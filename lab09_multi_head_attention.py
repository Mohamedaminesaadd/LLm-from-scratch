import math
from pathlib import Path
from collections import Counter
import re

from lab06_positional_encoding import PositionalEncoding, TextDataset, build_vocabulary, clean_text, encode, load_text
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


# -------------------------------------------------
# Build Vocabulary
# -------------------------------------------------


# -------------------------------------------------
# Encode Text
# -------------------------------------------------

# -------------------------------------------------
# Dataset
# -------------------------------------------------

# -------------------------------------------------
# Self Attention
# -------------------------------------------------



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
        attention = attention.transpose(1, 2).contiguous() # use for contiguous to make sure the tensor is change how is look in memory not just modified the stride 
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