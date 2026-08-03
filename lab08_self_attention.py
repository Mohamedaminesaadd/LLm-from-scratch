"""
=================================================
Lab 07 - Scaled Dot-Product Self-Attention
=================================================

Objective:
1. Load and prepare a text dataset.
2. Convert token IDs into embeddings.
3. Add positional encoding.
4. Create Query, Key, and Value.
5. Compute attention scores.
6. Scale the attention scores.
7. Apply softmax.
8. Compute the final attention output.

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
    """Load text from a file."""

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# -------------------------------------------------
# Clean Text
# -------------------------------------------------

def clean_text(text: str) -> str:
    """Clean and normalize text."""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    # Replace multiple spaces/newlines with one space
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# -------------------------------------------------
# Build Vocabulary
# -------------------------------------------------

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

    # Add special tokens first
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


# -------------------------------------------------
# Encode Text
# -------------------------------------------------

def encode(
    text: str,
    word_to_id: dict
) -> list[int]:
    """
    Convert text into token IDs.
    """

    tokens = text.split()

    ids = []

    unk_id = word_to_id["<UNK>"]

    for token in tokens:

        token_id = word_to_id.get(
            token,
            unk_id
        )

        ids.append(token_id)

    return ids


# -------------------------------------------------
# PyTorch Dataset
# -------------------------------------------------

class TextDataset(Dataset):
    """
    Create next-token prediction samples.

    Example:

    Token IDs:

        [10, 20, 30, 40, 50, 60]

    Input:

        [10, 20, 30, 40, 50]

    Target:

        [20, 30, 40, 50, 60]
    """

    def __init__(
        self,
        token_ids,
        sequence_length
    ):

        self.inputs = []
        self.targets = []

        for i in range(
            len(token_ids) - sequence_length
        ):

            # Input sequence
            x = token_ids[
                i:i + sequence_length
            ]

            # Target sequence shifted by one token
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
        """Return number of training samples."""

        return len(self.inputs)

    def __getitem__(self, index):
        """Return one input-target pair."""

        return (
            self.inputs[index],
            self.targets[index]
        )


# -------------------------------------------------
# Positional Encoding
# -------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Adds position information to token embeddings.
    """

    def __init__(
        self,
        embedding_dim,
        max_length=5000
    ):

        super().__init__()

        # -------------------------------------------------
        # Create positional encoding matrix
        #
        # Shape:
        #
        # [max_length, embedding_dim]
        # -------------------------------------------------

        pe = torch.zeros(
            max_length,
            embedding_dim
        )

        # -------------------------------------------------
        # Positions
        #
        # [0]
        # [1]
        # [2]
        # [3]
        # ...
        #
        # Shape:
        #
        # [max_length, 1]
        # -------------------------------------------------

        position = torch.arange(
            0,
            max_length,
            dtype=torch.float
        ).unsqueeze(1)

        # -------------------------------------------------
        # Frequency scaling
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Even dimensions -> sine
        # -------------------------------------------------

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        # -------------------------------------------------
        # Odd dimensions -> cosine
        # -------------------------------------------------

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # -------------------------------------------------
        # Add batch dimension
        #
        # Before:
        #
        # [max_length, embedding_dim]
        #
        # After:
        #
        # [1, max_length, embedding_dim]
        # -------------------------------------------------

        pe = pe.unsqueeze(0)

        # Positional encoding is fixed,
        # not trainable.
        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        # x shape:
        #
        # [batch_size, sequence_length, embedding_dim]

        sequence_length = x.size(1)

        # Add position information
        x = x + self.pe[
            :,
            :sequence_length,
            :
        ]

        return x


# -------------------------------------------------
# Scaled Dot-Product Self-Attention
# -------------------------------------------------

class SelfAttention(nn.Module):
    """
    Single-head scaled dot-product self-attention.

    Formula:

        Attention(Q, K, V)
            =
        softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(
        self,
        embedding_dim
    ):

        super().__init__()

        self.embedding_dim = embedding_dim

        # -------------------------------------------------
        # Query projection
        # -------------------------------------------------

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # -------------------------------------------------
        # Key projection
        # -------------------------------------------------

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # -------------------------------------------------
        # Value projection
        # -------------------------------------------------

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        # -------------------------------------------------
        # Create Query, Key, Value
        #
        # Input:
        #
        # x = [B, S, D]
        #
        # Output:
        #
        # Q = [B, S, D]
        # K = [B, S, D]
        # V = [B, S, D]
        # -------------------------------------------------

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        # -------------------------------------------------
        # Compute Attention Scores
        #
        # Q:
        #
        # [B, S, D]
        #
        # K.transpose:
        #
        # [B, D, S]
        #
        # Result:
        #
        # [B, S, S]
        # -------------------------------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        # -------------------------------------------------
        # Scale Attention Scores
        #
        # scores / sqrt(d_k)
        # -------------------------------------------------

        scores = scores / math.sqrt(
            self.embedding_dim
        )

        # -------------------------------------------------
        # Apply Softmax
        #
        # Convert scores into attention probabilities.
        #
        # Every row sums to 1.
        # -------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # -------------------------------------------------
        # Compute Attention Output
        #
        # AttentionWeights:
        #
        # [B, S, S]
        #
        # V:
        #
        # [B, S, D]
        #
        # Output:
        #
        # [B, S, D]
        # -------------------------------------------------

        output = torch.matmul(
            attention_weights,
            V
        )

        return (
            output,
            attention_weights
        )


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    # -------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------

    print("Loading dataset...")

    text = load_text(
        DATASET_PATH
    )

    # -------------------------------------------------
    # 2. Clean Dataset
    # -------------------------------------------------

    text = clean_text(text)

    # -------------------------------------------------
    # 3. Build Vocabulary
    # -------------------------------------------------

    word_to_id, id_to_word = (
        build_vocabulary(text)
    )

    # -------------------------------------------------
    # 4. Encode Text
    # -------------------------------------------------

    token_ids = encode(
        text,
        word_to_id
    )

    # -------------------------------------------------
    # 5. Create Dataset
    # -------------------------------------------------

    dataset = TextDataset(
        token_ids,
        SEQUENCE_LENGTH
    )

    # -------------------------------------------------
    # 6. Create DataLoader
    # -------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # -------------------------------------------------
    # 7. Create Embedding Layer
    # -------------------------------------------------

    embedding = nn.Embedding(
        num_embeddings=len(word_to_id),
        embedding_dim=EMBEDDING_DIM
    )

    # -------------------------------------------------
    # 8. Create Positional Encoding
    # -------------------------------------------------

    positional_encoding = PositionalEncoding(
        EMBEDDING_DIM
    )

    # -------------------------------------------------
    # 9. Create Self-Attention
    # -------------------------------------------------

    self_attention = SelfAttention(
        EMBEDDING_DIM
    )

    # -------------------------------------------------
    # Dataset Information
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Process One Batch
    # -------------------------------------------------

    for inputs, targets in dataloader:

        # -------------------------------------------------
        # Token IDs
        # -------------------------------------------------

        print("=" * 60)
        print("TOKEN IDs")
        print("=" * 60)

        print(inputs)

        print()

        # -------------------------------------------------
        # Token IDs -> Embeddings
        #
        # [B, S]
        #
        # ->
        #
        # [B, S, D]
        # -------------------------------------------------

        token_embeddings = embedding(
            inputs
        )

        # -------------------------------------------------
        # Add Positional Encoding
        # -------------------------------------------------

        embedded_inputs = positional_encoding(
            token_embeddings
        )

        # -------------------------------------------------
        # Self-Attention
        # -------------------------------------------------

        (
            attention_output,
            attention_weights
        ) = self_attention(
            embedded_inputs
        )

        # -------------------------------------------------
        # Shapes
        # -------------------------------------------------

        print("=" * 60)
        print("TENSOR SHAPES")
        print("=" * 60)

        print(
            "Input IDs           :",
            inputs.shape
        )

        print(
            "Token Embeddings    :",
            token_embeddings.shape
        )

        print(
            "Position + Embedding:",
            embedded_inputs.shape
        )

        print(
            "Attention Weights   :",
            attention_weights.shape
        )

        print(
            "Attention Output    :",
            attention_output.shape
        )

        print()

        # -------------------------------------------------
        # First Sequence
        # -------------------------------------------------

        first_sentence = inputs[0]

        print("=" * 60)
        print("TOKENS AND EMBEDDINGS")
        print("=" * 60)

        for position, token_id in enumerate(
            first_sentence
        ):

            idx = token_id.item()

            word = id_to_word[idx]

            vector = embedded_inputs[
                0,
                position
            ]

            print(
                f"{position:2d} | "
                f"{word:15} -> "
                f"{idx:3d} -> "
                f"{vector[:5]}"
            )

        print()

        # -------------------------------------------------
        # Attention Matrix
        # -------------------------------------------------

        first_attention = attention_weights[0]

        print("=" * 60)
        print("ATTENTION MATRIX")
        print("=" * 60)

        print(first_attention)

        print()

        # -------------------------------------------------
        # Verify Softmax
        # -------------------------------------------------

        print("=" * 60)
        print("SUM OF ATTENTION ROWS")
        print("=" * 60)

        print(
            first_attention.sum(
                dim=-1
            )
        )

        print()

        # -------------------------------------------------
        # Display Words
        # -------------------------------------------------

        words = [
            id_to_word[token_id.item()]
            for token_id in first_sentence
        ]

        print("=" * 60)
        print("WORDS")
        print("=" * 60)

        print(words)

        print()

        # -------------------------------------------------
        # Attention per Query
        # -------------------------------------------------

        print("=" * 60)
        print("ATTENTION PER TOKEN")
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

        # -------------------------------------------------
        # Only inspect one batch
        # -------------------------------------------------

        break


# -------------------------------------------------
# Run
# -------------------------------------------------

if __name__ == "__main__":
    main()