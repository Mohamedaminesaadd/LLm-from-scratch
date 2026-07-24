"""
=================================================
Lab 06 - Positional Encoding
=================================================

Objective:
1. Understand why positional encoding is needed.
2. Implement sinusoidal positional encoding.
3. Add positional information to token embeddings.
4. Visualize the resulting tensor shapes.
"""

from pathlib import Path
from collections import Counter
import re
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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

        print("=" * 50)
        print("Embedded Inputs + Positional Encoding")
        print("=" * 50)
        print(embedded_inputs)
        print()

        print("=" * 50)
        print("Shapes")
        print("=" * 50)
        print("Input IDs Shape        :", inputs.shape)
        print("Embedded Input Shape   :", embedded_inputs.shape)
        print("Target IDs Shape       :", targets.shape)
        print("Embedded Target Shape  :", embedded_targets.shape)
        print()

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