import math
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

from lab06_positional_encoding import (
    BATCH_SIZE,
    DATASET_PATH,
    EMBEDDING_DIM,
    SEQUENCE_LENGTH,
    PositionalEncoding,
    TextDataset,
    build_vocabulary,
    clean_text,
    encode,
    load_text,
)

from lab13_encoder_block import (
    DROPOUT,
    FF_HIDDEN_DIM,
    HEAD_NUMBER,
    AddNorm,
    EncoderBlock,
    FeedForward,
    MultiHeadSelfAttention,
)


# -------------------------------------------------
# Configuration
# -------------------------------------------------

NUM_LAYERS = 15


# -------------------------------------------------
# Encoder Block (one layer)
# -------------------------------------------------

# -------------------------------------------------
# Transformer Encoder (stack of N encoder blocks)
# -------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    Full encoder:
        embedding -> positional encoding -> dropout
        -> N x EncoderBlock
        -> final LayerNorm
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        head_number,
        hidden_dim,
        num_layers,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # The stack of identical (but independently-weighted) encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(embedding_dim, head_number, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final normalization (common in modern implementations)
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)

        # Scale embeddings by sqrt(d_model), as in the original paper
        x = self.embedding(token_ids) * math.sqrt(self.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        attention_maps = []

        for layer in self.layers:
            x, attention_weights = layer(x)
            attention_maps.append(attention_weights)

        x = self.final_norm(x)

        return x, attention_maps


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    # Load dataset
    text = load_text(DATASET_PATH)
    text = clean_text(text)

    # Vocabulary
    word_to_id, id_to_word = build_vocabulary(text)
    vocab_size = len(word_to_id)

    # Encode
    token_ids = encode(text, word_to_id)

    # Dataset + DataLoader
    dataset = TextDataset(token_ids, SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Full encoder = stack of NUM_LAYERS encoder blocks
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        head_number=HEAD_NUMBER,
        hidden_dim=FF_HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    encoder.eval()  # deterministic: disables dropout for inspection

    total_params = sum(p.numel() for p in encoder.parameters())

    print("=" * 50)
    print("Encoder summary")
    print("=" * 50)
    print(f"vocab_size     : {vocab_size}")
    print(f"embedding_dim  : {EMBEDDING_DIM}")
    print(f"head_number    : {HEAD_NUMBER}")
    print(f"ff_hidden_dim  : {FF_HIDDEN_DIM}")
    print(f"num_layers     : {NUM_LAYERS}")
    print(f"total params   : {total_params:,}")
    print()

    for inputs, targets in dataloader:

        print("=" * 50)
        print("Token IDs")
        print("=" * 50)
        print(inputs)
        print()

        # Forward through the full encoder stack
        encoder_output, attention_maps = encoder(inputs)

        print("=" * 50)
        print("Input Shape (batch, seq_len)")
        print("=" * 50)
        print(inputs.shape)
        print()

        print("=" * 50)
        print("Encoder Output Shape (batch, seq_len, d_model)")
        print("=" * 50)
        print(encoder_output.shape)
        print()

        print("=" * 50)
        print("Number of attention maps (= num_layers)")
        print("=" * 50)
        print(len(attention_maps))
        print("each map shape :", attention_maps[0].shape)  # (batch, heads, seq, seq)
        print()

        # Check final LayerNorm : mean ~ 0 and std ~ 1 on the last dimension
        print("=" * 50)
        print("Final LayerNorm Check (last dim)")
        print("=" * 50)
        print("mean :", encoder_output.mean(dim=-1))
        print("std  :", encoder_output.std(dim=-1))
        print()

        first_sentence = inputs[0]

        print("=" * 50)
        print("Tokens and Encoder Output")
        print("=" * 50)

        for i, token_id in enumerate(first_sentence):
            idx = token_id.item()
            word = id_to_word[idx]
            vector = encoder_output[0][i]
            print(f"{word:12} -> {idx:3d} -> {vector[:5]}")

        break


if __name__ == "__main__":
    main()