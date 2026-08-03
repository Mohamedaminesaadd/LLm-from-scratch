"""
=================================================
Lab 11 - Residual Connection
=================================================

Objective:
1. Understand residual connections.
2. Implement a residual block.
3. Preserve information from previous layers.
"""

import torch
import torch.nn as nn


# -------------------------------------------------
# Configuration
# -------------------------------------------------

BATCH_SIZE = 2
SEQUENCE_LENGTH = 5
EMBEDDING_DIM = 100
HIDDEN_DIM =   1024  # Typically 4 times the embedding dimension


# -------------------------------------------------
# Example Input
# -------------------------------------------------

x = torch.randn(
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    EMBEDDING_DIM
)


# -------------------------------------------------
# Feed Forward Network
# -------------------------------------------------

class FeedForward(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):

        return self.network(x)


# -------------------------------------------------
# Residual Block
# -------------------------------------------------

class ResidualConnection(nn.Module):

    def __init__(self, sublayer):

        super().__init__()

        self.sublayer = sublayer

    def forward(self, x):

        return x + self.sublayer(x)

def main():

    # -------------------------------------------------
    # Create Model
    # -------------------------------------------------

    feed_forward = FeedForward(
        EMBEDDING_DIM,
        HIDDEN_DIM
    )

    residual = ResidualConnection(feed_forward)


    # -------------------------------------------------
    # Forward Pass
    # -------------------------------------------------

    output = residual(x)


    # -------------------------------------------------
    # Display Results
    # -------------------------------------------------

    print("=" * 50)
    print("Input Shape")
    print("=" * 50)
    print(x.shape)

    print()

    print("=" * 50)
    print("Output Shape")
    print("=" * 50)
    print(output.shape)

    print()

    print("=" * 50)
    print("First Token")
    print("=" * 50)

    print("Input:")
    print(x[0, 0])

    print()

    print("Output:")
    print(output[0, 0])

if __name__ == "__main__":
    main()