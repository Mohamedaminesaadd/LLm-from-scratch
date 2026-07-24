"""
=================================================
Lab 07 - Scaled Dot-Product Attention
=================================================

Objective:
1. Understand Query, Key, and Value.
2. Compute attention scores.
3. Apply scaling.
4. Apply softmax.
5. Compute the final attention output.
"""

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

def build_vocabulary(text: str):

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
        ids.append(
            word_to_id.get(
                token,
                word_to_id["<UNK>"]
            )
        )

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

        position = torch.arange(
            0,
            max_length,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embedding_dim,
                2,
                dtype=torch.float
            ) * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        sequence_length = x.size(1)
        x = x + self.pe[:, :sequence_length]
        return x

# -------------------------------------------------
# Main
# -------------------------------------------------


def main():

    # Load and clean text
    text = load_text(DATASET_PATH)
    text = clean_text(text)

    # Build vocabulary
    word_to_id, id_to_word = build_vocabulary(text)

    # Encode text
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

    # Embedding layer
    embedding = nn.Embedding(
        num_embeddings=len(word_to_id),
        embedding_dim=EMBEDDING_DIM
    )

    # Positional Encoding
    positional_encoding = PositionalEncoding(EMBEDDING_DIM)

    for inputs, targets in dataloader:

        print("=" * 50)
        print("Token IDs")
        print("=" * 50)
        print(inputs)
        print()

        # Token Embeddings
        embedded_inputs = embedding(inputs)
        embedded_targets = embedding(targets)

        # Add Positional Encoding
        embedded_inputs = positional_encoding(embedded_inputs)
        embedded_targets = positional_encoding(embedded_targets)



        # -------------------------------------------------
        # Linear Layers
        # -------------------------------------------------

        query_layer = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        key_layer = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        value_layer = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

        # -------------------------------------------------
        # Create Q, K, V
        # -------------------------------------------------

        Q = query_layer(embedded_inputs)
        K = key_layer(embedded_inputs)
        V = value_layer(embedded_inputs)
        # -------------------------------------------------
        # Compute Attention Scores
        # -------------------------------------------------

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(EMBEDDING_DIM)


        # -------------------------------------------------
        # Apply Softmax
        # -------------------------------------------------

        attention_weights = torch.softmax(scores, dim=-1)


        # -------------------------------------------------
        # Compute Final Output
        # -------------------------------------------------

        output = torch.matmul(attention_weights, V)


        # -------------------------------------------------
        # Display Results
        # -------------------------------------------------

        print("=" * 50)
        print("Input Shape")
        print("=" * 50)
        print(embedded_inputs.shape)

        print()

        print("=" * 50)
        print("Query Shape")
        print("=" * 50)
        print(Q.shape)

        print()

        print("=" * 50)
        print("Key Shape")
        print("=" * 50)
        print(K.shape)

        print()

        print("=" * 50)
        print("Value Shape")
        print("=" * 50)
        print(V.shape)

        print()

        print("=" * 50)
        print("Attention Scores Shape")
        print("=" * 50)
        print(scores.shape)

        print()

        print("=" * 50)
        print("Attention Weights Shape")
        print("=" * 50)
        print(attention_weights.shape)

        print()

        print("=" * 50)
        print("Output Shape")
        print("=" * 50)
        print(output.shape)
        print("=" * 50)
        print("First Sentence")
        print("=" * 50)

        first_sentence = inputs[0]

        for token_id in first_sentence:
            idx = token_id.item()
            word = id_to_word[idx]
            vector = embedded_inputs[0][
                (first_sentence == token_id).nonzero(as_tuple=True)[0][0]
            ]

            print(f"{word:12} -> {idx:3d} -> {vector[:5]} ...")

        break


if __name__ == "__main__":
    main()