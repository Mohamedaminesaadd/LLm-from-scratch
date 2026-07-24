import math
from pathlib import Path
from collections import Counter
import re

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

import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embedding_dim, head_number):
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

    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)

        batch_size, seq_len, _ = x.size()

        # Compute Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into heads
        Q = Q.view(batch_size, seq_len, self.head_number, self.head_dim)
        K = K.view(batch_size, seq_len, self.head_number, self.head_dim)
        V = V.view(batch_size, seq_len, self.head_number, self.head_dim)

        # Move head dimension before sequence
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # Attention output
        attention = torch.matmul(attention_weights, V)

        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, seq_len, self.embedding_dim)

        # Final linear layer
        output = self.out(attention)

        return output, attention_weights

    
# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    # Load dataset
    text = load_text(DATASET_PATH)
    text = clean_text(text)

    # Vocabulary
    word_to_id, id_to_word = build_vocabulary(text)

    # Encode
    token_ids = encode(text, word_to_id)

    # Dataset
    dataset = TextDataset(
        token_ids,
        SEQUENCE_LENGTH
    )

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Embedding
    embedding = nn.Embedding(
        num_embeddings=len(word_to_id),
        embedding_dim=EMBEDDING_DIM
    )

    # Positional Encoding
    positional_encoding = PositionalEncoding(EMBEDDING_DIM)

    # Self Attention
    multi_head_attention = MultiHeadSelfAttention(
    embedding_dim=EMBEDDING_DIM,
    head_number=4)

    for inputs, targets in dataloader:

        print("=" * 50)
        print("Token IDs")
        print("=" * 50)
        print(inputs)
        print()

        # Embeddings
        embedded_inputs = embedding(inputs)
        embedded_targets = embedding(targets)

        # Positional Encoding
        embedded_inputs = positional_encoding(embedded_inputs)
        embedded_targets = positional_encoding(embedded_targets)

        # Self Attention
        attention_output, attention_weights = multi_head_attention(embedded_inputs)

        print("=" * 50)
        print("Input Shape")
        print("=" * 50)
        print(embedded_inputs.shape)
        print()

        print("=" * 50)
        print("Attention Output Shape")
        print("=" * 50)
        print(attention_output.shape)
        print()

        print("=" * 50)
        print("Attention Weights Shape")
        print("=" * 50)
        print(attention_weights.shape)
        print()

        print("=" * 50)
        print("Attention Weights")
        print("=" * 50)
        print(attention_weights)
        print()

        first_sentence = inputs[0]

        print("=" * 50)
        print("Tokens and Embeddings")
        print("=" * 50)

        for i, token_id in enumerate(first_sentence):
            idx = token_id.item()
            word = id_to_word[idx]
            vector = embedded_inputs[0][i]

            print(f"{word:12} -> {idx:3d} -> {vector[:5]}")

        break


if __name__ == "__main__":
    main()