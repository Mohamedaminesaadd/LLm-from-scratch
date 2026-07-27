import math
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATASET_PATH = Path("datasets/sample.txt")

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
]

SEQUENCE_LENGTH = 5
BATCH_SIZE = 2
EMBEDDING_DIM = 100
HEAD_NUMBER = 4
FF_HIDDEN_DIM = 4 * EMBEDDING_DIM
DROPOUT = 0.1
NUM_LAYERS = 12          # <-- 12 encoder blocks stacked


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------------------------------
# Build Vocabulary
# -------------------------------------------------

def build_vocabulary(text):

    counter = Counter(text.split())

    word_to_id = {}
    id_to_word = {}

    idx = 0

    for token in SPECIAL_TOKENS:
        word_to_id[token] = idx
        id_to_word[idx] = token
        idx += 1

    for word in sorted(counter.keys()):
        word_to_id[word] = idx
        id_to_word[idx] = word
        idx += 1

    return word_to_id, id_to_word


# -------------------------------------------------
# Encode Text
# -------------------------------------------------

def encode(text, word_to_id):

    tokens = text.split()

    ids = []

    for token in tokens:
        ids.append(word_to_id.get(token, word_to_id["<UNK>"]))

    return ids


# -------------------------------------------------
# Dataset
# -------------------------------------------------

class TextDataset(Dataset):

    def __init__(self, token_ids, sequence_length):

        self.inputs = []
        self.targets = []

        for i in range(len(token_ids) - sequence_length):

            x = token_ids[i:i + sequence_length]
            y = token_ids[i + 1:i + sequence_length + 1]

            self.inputs.append(torch.tensor(x, dtype=torch.long))
            self.targets.append(torch.tensor(y, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


# -------------------------------------------------
# Positional Encoding
# -------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim, max_length=5000):
        super().__init__()

        pe = torch.zeros(max_length, embedding_dim)

        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]


# -------------------------------------------------
# Self Attention
# -------------------------------------------------

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embedding_dim, head_number, dropout=0.1):
        super().__init__()

        assert embedding_dim % head_number == 0, \
            "embedding_dim must be divisible by head_number"

        self.embedding_dim = embedding_dim
        self.head_number = head_number
        self.head_dim = embedding_dim // head_number

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, embedding_dim)

        batch_size, seq_len, _ = x.size()

        # Compute Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into heads -> (batch, heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.head_number, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_number, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_number, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Optional mask (padding or causal)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax + dropout on the weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Attention output
        attention = torch.matmul(attention_weights, V)

        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, seq_len, self.embedding_dim)

        # Final linear layer
        output = self.out(attention)

        return output, attention_weights


# -------------------------------------------------
# Feed Forward Network (position-wise)
# -------------------------------------------------

class FeedForward(nn.Module):
    """
    Applied independently to each position:
        Linear(d_model -> d_ff) -> ReLU -> Dropout -> Linear(d_ff -> d_model)
    """

    def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


# -------------------------------------------------
# Residual Connection + Layer Normalization
# -------------------------------------------------

class AddNorm(nn.Module):
    """
    Post-norm (original paper):  LayerNorm(x + Dropout(sublayer(x)))
    """

    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))


# -------------------------------------------------
# Encoder Block (one layer)
# -------------------------------------------------

class EncoderBlock(nn.Module):
    """
    One encoder layer:
        x -> MultiHeadSelfAttention -> AddNorm -> FeedForward -> AddNorm
    """

    def __init__(self, embedding_dim, head_number, hidden_dim, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(embedding_dim, head_number, dropout)
        self.add_norm_1 = AddNorm(embedding_dim, dropout)

        self.feed_forward = FeedForward(embedding_dim, hidden_dim, dropout)
        self.add_norm_2 = AddNorm(embedding_dim, dropout)

    def forward(self, x, mask=None):

        # Sub-layer 1 : self-attention + residual + norm
        attention_output, attention_weights = self.attention(x, mask)
        x = self.add_norm_1(x, attention_output)

        # Sub-layer 2 : feed forward + residual + norm
        feed_forward_output = self.feed_forward(x)
        x = self.add_norm_2(x, feed_forward_output)

        return x, attention_weights


# -------------------------------------------------
# Transformer Encoder (stack of N encoder blocks)
# -------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Full encoder:
        embedding -> positional encoding -> dropout
        -> N x EncoderBlock
        -> final LayerNorm
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        head_number,
        hidden_dim,
        num_layers,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # The stack of identical (but independently-weighted) encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(embedding_dim, head_number, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final normalization (common in modern implementations)
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, token_ids, mask=None):
        # token_ids: (batch_size, seq_len)

        # Scale embeddings by sqrt(d_model), as in the original paper
        x = self.embedding(token_ids) * math.sqrt(self.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        attention_maps = []

        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            attention_maps.append(attention_weights)

        x = self.final_norm(x)

        return x, attention_maps


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    # Load dataset
    text = load_text(DATASET_PATH)
    text = clean_text(text)

    # Vocabulary
    word_to_id, id_to_word = build_vocabulary(text)
    vocab_size = len(word_to_id)

    # Encode
    token_ids = encode(text, word_to_id)

    # Dataset + DataLoader
    dataset = TextDataset(token_ids, SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Full encoder = stack of 12 encoder blocks
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        head_number=HEAD_NUMBER,
        hidden_dim=FF_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    encoder.eval()  # deterministic: disables dropout for inspection

    total_params = sum(p.numel() for p in encoder.parameters())

    print("=" * 50)
    print("Encoder summary")
    print("=" * 50)
    print(f"vocab_size     : {vocab_size}")
    print(f"embedding_dim  : {EMBEDDING_DIM}")
    print(f"head_number    : {HEAD_NUMBER}")
    print(f"ff_hidden_dim  : {FF_HIDDEN_DIM}")
    print(f"num_layers     : {NUM_LAYERS}")
    print(f"total params   : {total_params:,}")
    print()

    for inputs, targets in dataloader:

        print("=" * 50)
        print("Token IDs")
        print("=" * 50)
        print(inputs)
        print()

        # Forward through the full 12-layer encoder
        encoder_output, attention_maps = encoder(inputs)

        print("=" * 50)
        print("Input Shape (batch, seq_len)")
        print("=" * 50)
        print(inputs.shape)
        print()

        print("=" * 50)
        print("Encoder Output Shape (batch, seq_len, d_model)")
        print("=" * 50)
        print(encoder_output.shape)
        print()

        print("=" * 50)
        print("Number of attention maps (= num_layers)")
        print("=" * 50)
        print(len(attention_maps))
        print("each map shape :", attention_maps[0].shape)  # (batch, heads, seq, seq)
        print()

        # Check final LayerNorm : mean ~ 0 and std ~ 1 on the last dimension
        print("=" * 50)
        print("Final LayerNorm Check (last dim)")
        print("=" * 50)
        print("mean :", encoder_output.mean(dim=-1))
        print("std  :", encoder_output.std(dim=-1))
        print()

        first_sentence = inputs[0]

        print("=" * 50)
        print("Tokens and Encoder Output")
        print("=" * 50)

        for i, token_id in enumerate(first_sentence):
            idx = token_id.item()
            word = id_to_word[idx]
            vector = encoder_output[0][i]
            print(f"{word:12} -> {idx:3d} -> {vector[:5]}")

        break


if __name__ == "__main__":
    main()