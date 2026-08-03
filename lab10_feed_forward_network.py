"""
=================================================
Lab 10 - Feed Forward Network
=================================================

Objective:
1. Understand the Feed Forward Network (FFN).
2. Build a two-layer neural network.
3. Apply the FFN independently to each token.
4. Explore input and output shapes.
"""

import torch
import torch.nn as nn


# -------------------------------------------------
# Configuration
# -------------------------------------------------

BATCH_SIZE = 2
SEQUENCE_LENGTH = 5
EMBEDDING_DIM = 100 
HIDDEN_DIM = 1024  # Typically 4 times the embedding dimension


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

        self.linear1 = nn.Linear(
            embedding_dim,
            hidden_dim
        )

        self.activation = nn.GELU()

        self.linear2 = nn.Linear(
            hidden_dim,
            embedding_dim
        )

    def forward(self, x):

        x = self.linear1(x)

        x = self.activation(x)

        x = self.linear2(x)

        return x

def main():

    # -------------------------------------------------
    # Create Model
    # -------------------------------------------------

    ffn = FeedForward(
        EMBEDDING_DIM,
        HIDDEN_DIM
    )


    # -------------------------------------------------
    # Forward Pass
    # -------------------------------------------------

    output = ffn(x)


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
    print("First Token Before FFN")
    print("=" * 50)
    print(x[0, 0])

    print()

    print("=" * 50)
    print("First Token After FFN")
    print("=" * 50)
    print(output[0, 0])


if __name__ == "__main__":
    main()