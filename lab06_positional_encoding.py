"""
=================================================
Lab 06 - Positional Encoding
=================================================

Objective:
1. Load and clean a text dataset.
2. Build a vocabulary.
3. Encode text into token IDs.
4. Create input-target training sequences.
5. Convert token IDs into embeddings.
6. Implement sinusoidal positional encoding.
7. Add positional information to embeddings.
8. Inspect tensor shapes.
"""

from pathlib import Path
from collections import Counter
import math
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =================================================
# Configuration
# =================================================

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
MAX_SEQUENCE_LENGTH = 5000


# =================================================
# Load Dataset
# =================================================

def load_text(path: Path) -> str:
    """Load text from a file."""

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# =================================================
# Clean Text
# =================================================

def clean_text(text: str) -> str:
    """Clean and normalize text."""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation/special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Replace multiple whitespaces with one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =================================================
# Build Vocabulary
# =================================================

def build_vocabulary(text: str):
    """
    Build:
        word -> ID
        ID   -> word
    """

    tokens = text.split()

    counter = Counter(tokens)

    word_to_id = {}
    id_to_word = {}

    idx = 0

    # ---------------------------------------------
    # Add special tokens first
    # ---------------------------------------------

    for token in SPECIAL_TOKENS:

        word_to_id[token] = idx
        id_to_word[idx] = token

        idx += 1

    # ---------------------------------------------
    # Add normal vocabulary words
    # ---------------------------------------------

    for word in sorted(counter.keys()):

        word_to_id[word] = idx
        id_to_word[idx] = word

        idx += 1

    return word_to_id, id_to_word


# =================================================
# Encode Text
# =================================================

def encode(text: str, word_to_id: dict) -> list[int]:
    """
    Convert text into token IDs.
    """

    tokens = text.split()

    unk_id = word_to_id["<UNK>"]

    token_ids = []

    for token in tokens:

        token_id = word_to_id.get(
            token,
            unk_id
        )

        token_ids.append(token_id)

    return token_ids


# =================================================
# PyTorch Dataset
# =================================================

class TextDataset(Dataset):
    """
    Create next-token prediction training samples.

    Example:

    tokens:
        [10, 20, 30, 40, 50, 60]

    input:
        [10, 20, 30, 40, 50]

    target:
        [20, 30, 40, 50, 60]
    """

    def __init__(
        self,
        token_ids: list[int],
        sequence_length: int
    ):

        self.inputs = []
        self.targets = []

        for i in range(
            len(token_ids) - sequence_length
        ):

            # Current sequence
            x = token_ids[
                i:i + sequence_length
            ]

            # Same sequence shifted by one token
            y = token_ids[
                i + 1:i + sequence_length + 1
            ]

            self.inputs.append(
                torch.tensor(
                    x,
                    dtype=torch.long
                )
            )

            self.targets.append(
                torch.tensor(
                    y,
                    dtype=torch.long
                )
            )

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        return (
            self.inputs[index],
            self.targets[index]
        )


# =================================================
# Sinusoidal Positional Encoding
# =================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))

    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(
        self,
        embedding_dim: int,
        max_length: int = 5000
    ):

        super().__init__()

        # -----------------------------------------
        # Create empty positional encoding matrix
        #
        # Shape:
        # [max_length, embedding_dim]
        # -----------------------------------------

        pe = torch.zeros(
            max_length,
            embedding_dim
        )

        # -----------------------------------------
        # Create positions
        #
        # [0, 1, 2, 3, ...]
        #
        # Shape:
        # [max_length, 1]
        # -----------------------------------------

        position = torch.arange(
            0,
            max_length,
            dtype=torch.float
        ).unsqueeze(1)

        # -----------------------------------------
        # Create frequency terms
        #
        # dimensions:
        #
        # 0, 2, 4, 6, ...
        #
        # Shape:
        # [embedding_dim / 2]
        # -----------------------------------------

        dimensions = torch.arange(
            0,
            embedding_dim,
            2,
            dtype=torch.float
        )

        div_term = torch.exp(
            dimensions
            * (
                -math.log(10000.0)
                / embedding_dim
            )
        )

        # -----------------------------------------
        # Even dimensions → sine
        # -----------------------------------------

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # -----------------------------------------
        # Odd dimensions → cosine
        # -----------------------------------------

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # -----------------------------------------
        # Add batch dimension
        #
        # Before:
        # [max_length, embedding_dim]
        #
        # After:
        # [1, max_length, embedding_dim]
        # -----------------------------------------

        pe = pe.unsqueeze(0)

        # Positional encoding is NOT trainable.
        # It is stored as part of the module.
        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        # x shape:
        #
        # [batch_size,
        #  sequence_length,
        #  embedding_dim]

        sequence_length = x.size(1)

        # Select only required positions
        position_vectors = self.pe[
            :,
            :sequence_length,
            :
        ]

        # Token embedding + position information
        x = x + position_vectors

        return x


# =================================================
# Main
# =================================================

def main():

    # ---------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------

    print("Loading dataset...")

    text = load_text(
        DATASET_PATH
    )

    # ---------------------------------------------
    # 2. Clean dataset
    # ---------------------------------------------

    text = clean_text(text)

    # ---------------------------------------------
    # 3. Build vocabulary
    # ---------------------------------------------

    word_to_id, id_to_word = (
        build_vocabulary(text)
    )

    # ---------------------------------------------
    # 4. Encode dataset
    # ---------------------------------------------

    token_ids = encode(
        text,
        word_to_id
    )

    # ---------------------------------------------
    # 5. Create Dataset
    # ---------------------------------------------

    dataset = TextDataset(
        token_ids,
        SEQUENCE_LENGTH
    )

    # ---------------------------------------------
    # 6. Create DataLoader
    # ---------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ---------------------------------------------
    # 7. Create Embedding Layer
    # ---------------------------------------------

    embedding = nn.Embedding(
        num_embeddings=len(word_to_id),
        embedding_dim=EMBEDDING_DIM
    )

    # ---------------------------------------------
    # 8. Create Positional Encoding
    # ---------------------------------------------

    positional_encoding = PositionalEncoding(
        embedding_dim=EMBEDDING_DIM,
        max_length=MAX_SEQUENCE_LENGTH
    )

    # =============================================
    # Dataset Information
    # =============================================

    print()

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(
        "Vocabulary Size :",
        len(word_to_id)
    )

    print(
        "Total Tokens    :",
        len(token_ids)
    )

    print(
        "Training Samples:",
        len(dataset)
    )

    print()

    # =============================================
    # Embedding Information
    # =============================================

    print("=" * 60)
    print("EMBEDDING INFORMATION")
    print("=" * 60)

    print(
        "Embedding Matrix Shape:",
        embedding.weight.shape
    )

    print(
        "Positional Encoding Shape:",
        positional_encoding.pe.shape
    )

    print()

    # =============================================
    # Process One Batch
    # =============================================

    for inputs, targets in dataloader:

        # -----------------------------------------
        # Token IDs
        # -----------------------------------------

        print("=" * 60)
        print("TOKEN IDs")
        print("=" * 60)

        print(inputs)

        print()

        # -----------------------------------------
        # Convert IDs → embeddings
        # -----------------------------------------

        token_embeddings = embedding(
            inputs
        )

        # -----------------------------------------
        # Add positional information
        # -----------------------------------------

        final_embeddings = (
            positional_encoding(
                token_embeddings
            )
        )

        # =========================================
        # Shapes
        # =========================================

        print("=" * 60)
        print("TENSOR SHAPES")
        print("=" * 60)

        print(
            "Input IDs:"
        )

        print(
            inputs.shape
        )

        print()

        print(
            "Token Embeddings:"
        )

        print(
            token_embeddings.shape
        )

        print()

        print(
            "Final Embeddings:"
        )

        print(
            final_embeddings.shape
        )

        print()

        # =========================================
        # First Sequence
        # =========================================

        print("=" * 60)
        print("FIRST SEQUENCE")
        print("=" * 60)

        first_sequence = inputs[0]

        for position, token_id in enumerate(
            first_sequence
        ):

            # Tensor → Python integer
            idx = token_id.item()

            # ID → word
            word = id_to_word[idx]

            # Original token embedding
            token_vector = (
                token_embeddings[
                    0,
                    position
                ]
            )

            # Position vector
            position_vector = (
                positional_encoding.pe[
                    0,
                    position
                ]
            )

            # Final vector
            final_vector = (
                final_embeddings[
                    0,
                    position
                ]
            )

            print()

            print(
                f"Position : {position}"
            )

            print(
                f"Word     : {word}"
            )

            print(
                f"Token ID : {idx}"
            )

            print(
                "Embedding :",
                token_vector[:5]
            )

            print(
                "Position  :",
                position_vector[:5]
            )

            print(
                "Final     :",
                final_vector[:5]
            )

        # Only inspect one batch
        break


# =================================================
# Run
# =================================================

if __name__ == "__main__":
    main()