"""
=================================================
Lab 05 - Embedding Layer
=================================================

Objective:
1. Understand token embeddings.
2. Create an nn.Embedding layer.
3. Convert token IDs into dense vectors.
4. Explore embedding dimensions.
"""

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
# Main
# -------------------------------------------------

def main():

    # Load text
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

    # Embedding Layer
    embedding = nn.Embedding(
        num_embeddings=len(word_to_id),
        embedding_dim=EMBEDDING_DIM
    )

    # -------------------------------------------------
    # Information
    # -------------------------------------------------

    print("=" * 50)
    print("Dataset Information")
    print("=" * 50)

    print("Vocabulary Size :", len(word_to_id))
    print("Total Tokens    :", len(token_ids))
    print("Training Samples:", len(dataset))

    print()

    print("=" * 50)
    print("Embedding Matrix")
    print("=" * 50)

    print(embedding.weight.shape)

    print()

    # -------------------------------------------------
    # One Batch
    # -------------------------------------------------

    for inputs, targets in dataloader:

        print("=" * 50)
        print("Token IDs")
        print("=" * 50)

        print(inputs)

        print()

        # Apply embedding
        embedded_inputs = embedding(inputs)
        embedded_targets = embedding(targets)

        print("=" * 50)
        print("Embedded Inputs")
        print("=" * 50)

        print(embedded_inputs)

        print()

        print("=" * 50)
        print("Shapes")
        print("=" * 50)

        print("Input IDs Shape       :", inputs.shape)
        print("Embedded Input Shape  :", embedded_inputs.shape)
        print("Target IDs Shape      :", targets.shape) # during the training we don't embedding the training datasets
        print("Embedded Target Shape :", embedded_targets.shape)

        print()

        print("=" * 50)
        print("Word -> ID -> Embedding")
        print("=" * 50)

        first_sentence = inputs[0]

        for token_id in first_sentence:

            idx = token_id.item()
            word = id_to_word[idx]
            vector = embedding.weight[idx]

            print(f"{word:10} -> {idx:2d} -> {vector[:5]} ...")

        break


if __name__ == "__main__":
    main()