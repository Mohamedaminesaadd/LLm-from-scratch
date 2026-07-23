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

import torch
import torch.nn as nn

# -------------------------------------------------
# Configuration
# -------------------------------------------------

VOCAB_SIZE = 20
EMBEDDING_DIM = 8


# Example batch of token IDs
#
# Batch Size = 2
# Sequence Length = 5
#
# Sentence 1 : [2, 5, 7, 9, 3]
# Sentence 2 : [2, 4, 6, 8, 3]
#


token_ids = torch.tensor([
    [2, 5, 7, 9, 3],
    [2, 4, 6, 8, 3]
])


# -------------------------------------------------
# Embedding Layer
# -------------------------------------------------

embedding = nn.Embedding(
    num_embeddings=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM
)


# -------------------------------------------------
# Forward Pass
# -------------------------------------------------

embedded_tokens = embedding(token_ids)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("=" * 50)
print("Token IDs")
print("=" * 50)

print(token_ids)

print()

print("=" * 50)
print("Embedding Weight Matrix")
print("=" * 50)

print(embedding.weight.shape)

print()

print("=" * 50)
print("Embedded Tokens")
print("=" * 50)

print(embedded_tokens)

print()

print("=" * 50)
print("Output Shape")
print("=" * 50)

print(embedded_tokens.shape)