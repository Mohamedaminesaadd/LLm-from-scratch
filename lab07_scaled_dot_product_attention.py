"""
=================================================
Lab 07 - Scaled Dot-Product Attention
=================================================

Objective:
1. Understand Query, Key, and Value.
2. Create Q, K, and V projections.
3. Compute attention scores.
4. Apply scaling.
5. Apply softmax.
6. Compute the final attention output.

Formula:

    Attention(Q, K, V)
        = softmax(QK^T / sqrt(d_k)) V
"""

import math
import re
from pathlib import Path
from collections import Counter

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

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =================================================
# Build Vocabulary
# =================================================

def build_vocabulary(text: str):

    tokens = text.split()

    counter = Counter(tokens)

    word_to_id = {}
    id_to_word = {}

    idx = 0

    # Add special tokens
    for token in SPECIAL_TOKENS:

        word_to_id[token] = idx
        id_to_word[idx] = token

        idx += 1

    # Add normal words
    for word in sorted(counter.keys()):

        word_to_id[word] = idx
        id_to_word[idx] = word

        idx += 1

    return word_to_id, id_to_word


# =================================================
# Encode Text
# =================================================

def encode(
    text: str,
    word_to_id: dict
) -> list[int]:

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

            x = token_ids[
                i:i + sequence_length
            ]

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
# Positional Encoding
# =================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        max_length: int = 5000
    ):

        super().__init__()

        # -----------------------------------------
        # Positional encoding matrix
        #
        # [max_length, embedding_dim]
        # -----------------------------------------

        pe = torch.zeros(
            max_length,
            embedding_dim
        )

        # -----------------------------------------
        # Positions
        #
        # [max_length, 1]
        # -----------------------------------------

        position = torch.arange(
            0,
            max_length,
            dtype=torch.float
        ).unsqueeze(1)

        # -----------------------------------------
        # Frequency terms
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

        # Even dimensions -> sine
        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # Odd dimensions -> cosine
        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Add batch dimension
        #
        # [1, max_length, embedding_dim]

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        sequence_length = x.size(1)

        position_vectors = self.pe[
            :,
            :sequence_length,
            :
        ]

        return x + position_vectors


# =================================================
# Scaled Dot-Product Self-Attention
# =================================================

class SelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int
    ):

        super().__init__()

        self.embedding_dim = embedding_dim

        # -----------------------------------------
        # Query projection
        # -----------------------------------------

        self.query_layer = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # -----------------------------------------
        # Key projection
        # -----------------------------------------

        self.key_layer = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # -----------------------------------------
        # Value projection
        # -----------------------------------------

        self.value_layer = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        # -----------------------------------------
        # Create Q, K, V
        #
        # x:
        # [batch, sequence, embedding]
        # -----------------------------------------

        Q = self.query_layer(x)

        K = self.key_layer(x)

        V = self.value_layer(x)

        # -----------------------------------------
        # Attention Scores
        #
        # Q @ K^T
        #
        # [B,S,D] @ [B,D,S]
        #
        # ->
        #
        # [B,S,S]
        # -----------------------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        # -----------------------------------------
        # Scale Scores
        #
        # QK^T / sqrt(d_k)
        # -----------------------------------------

        scores = scores / math.sqrt(
            self.embedding_dim
        )

        # -----------------------------------------
        # Softmax
        #
        # Convert scores into attention weights
        # -----------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # -----------------------------------------
        # Attention Output
        #
        # AttentionWeights @ V
        #
        # [B,S,S] @ [B,S,D]
        #
        # ->
        #
        # [B,S,D]
        # -----------------------------------------

        output = torch.matmul(
            attention_weights,
            V
        )

        return (
            output,
            attention_weights,
            Q,
            K,
            V
        )


# =================================================
# Main
# =================================================

def main():

    # ---------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------

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
    # 4. Encode text
    # ---------------------------------------------

    token_ids = encode(
        text,
        word_to_id
    )

    # ---------------------------------------------
    # 5. Dataset
    # ---------------------------------------------

    dataset = TextDataset(
        token_ids,
        SEQUENCE_LENGTH
    )

    # ---------------------------------------------
    # 6. DataLoader
    # ---------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ---------------------------------------------
    # 7. Embedding
    # ---------------------------------------------

    embedding = nn.Embedding(
        num_embeddings=len(word_to_id),
        embedding_dim=EMBEDDING_DIM
    )

    # ---------------------------------------------
    # 8. Positional Encoding
    # ---------------------------------------------

    positional_encoding = PositionalEncoding(
        embedding_dim=EMBEDDING_DIM,
        max_length=MAX_SEQUENCE_LENGTH
    )

    # ---------------------------------------------
    # 9. Self-Attention
    # ---------------------------------------------

    attention = SelfAttention(
        embedding_dim=EMBEDDING_DIM
    )

    # =================================================
    # Information
    # =================================================

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

    # =================================================
    # Process One Batch
    # =================================================

    for inputs, targets in dataloader:

        print("=" * 60)
        print("TOKEN IDs")
        print("=" * 60)

        print(inputs)

        print()

        # -----------------------------------------
        # Token IDs -> Embeddings
        # -----------------------------------------

        token_embeddings = embedding(
            inputs
        )

        # -----------------------------------------
        # Add positional information
        # -----------------------------------------

        x = positional_encoding(
            token_embeddings
        )

        # -----------------------------------------
        # Self-Attention
        # -----------------------------------------

        (
            output,
            attention_weights,
            Q,
            K,
            V
        ) = attention(x)

        # =================================================
        # Shapes
        # =================================================

        print("=" * 60)
        print("TENSOR SHAPES")
        print("=" * 60)

        print(
            "Input IDs            :",
            inputs.shape
        )

        print(
            "Embeddings           :",
            token_embeddings.shape
        )

        print(
            "Position-aware Input :",
            x.shape
        )

        print(
            "Query                :",
            Q.shape
        )

        print(
            "Key                  :",
            K.shape
        )

        print(
            "Value                :",
            V.shape
        )

        print(
            "Attention Weights    :",
            attention_weights.shape
        )

        print(
            "Attention Output     :",
            output.shape
        )

        print()

        # =================================================
        # First Sequence
        # =================================================

        print("=" * 60)
        print("FIRST SEQUENCE")
        print("=" * 60)

        first_sequence = inputs[0]

        for position, token_id in enumerate(
            first_sequence
        ):

            idx = token_id.item()

            word = id_to_word[idx]

            vector = x[
                0,
                position
            ]

            print(
                f"{position:2d} | "
                f"{word:15} | "
                f"ID={idx:4d} | "
                f"{vector[:5]}"
            )

        print()

        # =================================================
        # Attention Matrix
        # =================================================

        print("=" * 60)
        print("ATTENTION MATRIX")
        print("=" * 60)

        first_attention = (
            attention_weights[0]
        )

        print(first_attention)

        print()

        # =================================================
        # Verify Softmax
        # =================================================

        print("=" * 60)
        print("SUM OF ATTENTION ROWS")
        print("=" * 60)

        print(
            first_attention.sum(dim=-1)
        )

        print()

        # =================================================
        # Words
        # =================================================

        words = [
            id_to_word[token_id.item()]
            for token_id in first_sequence
        ]

        print(
            "Words:",
            words
        )

        print()

        # =================================================
        # Attention per Word
        # =================================================

        print("=" * 60)
        print("ATTENTION PER WORD")
        print("=" * 60)

        for query_position, query_word in enumerate(
            words
        ):

            print(
                f"\nQuery: {query_word}"
            )

            weights = first_attention[
                query_position
            ]

            for key_position, key_word in enumerate(
                words
            ):

                weight = weights[
                    key_position
                ].item()

                print(
                    f"    {key_word:15}"
                    f" -> {weight:.4f}"
                )

        # Only inspect one batch
        break


# =================================================
# Run
# =================================================

if __name__ == "__main__":
    main()