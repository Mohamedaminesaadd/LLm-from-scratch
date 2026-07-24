"""
=================================================
Lab 12 - Layer Normalization
=================================================

Objective:
1. Understand Layer Normalization.
2. Normalize token embeddings.
3. Learn why LayerNorm is used in Transformers.
"""

import torch
import torch.nn as nn


# -------------------------------------------------
# Configuration
# -------------------------------------------------

BATCH_SIZE = 2
SEQUENCE_LENGTH = 5
EMBEDDING_DIM = 16


# -------------------------------------------------
# Example Input
# -------------------------------------------------

x = torch.randn(
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    EMBEDDING_DIM
)


# -------------------------------------------------
# Layer Normalization
# -------------------------------------------------

layer_norm = nn.LayerNorm(EMBEDDING_DIM)


# -------------------------------------------------
# Forward Pass
# -------------------------------------------------

output = layer_norm(x)


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
print("First Token Before LayerNorm")
print("=" * 50)
print(x[0, 0])

print()

print("=" * 50)
print("First Token After LayerNorm")
print("=" * 50)
print(output[0, 0])

print()

print("=" * 50)
print("Mean After LayerNorm")
print("=" * 50)
print(output[0, 0].mean())

print()

print("=" * 50)
print("Standard Deviation After LayerNorm")
print("=" * 50)
print(output[0, 0].std(unbiased=False))