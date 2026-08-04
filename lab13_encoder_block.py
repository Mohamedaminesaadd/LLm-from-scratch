import math
import re
from collections import Counter
from pathlib import Path

from lab06_positional_encoding import PositionalEncoding, TextDataset, build_vocabulary, clean_text, encode, load_text
from lab09_multi_head_attention import MultiHeadSelfAttention
from lab10_feed_forward_network import FeedForward
from lab11_residual_connection import AddNorm
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
HEAD_NUMBER = 4
FF_HIDDEN_DIM = 4 * EMBEDDING_DIM
DROPOUT = 0.1



class EncoderBlock(nn.Module):
    """
    x -> MultiHeadSelfAttention -> AddNorm -> FeedForward -> AddNorm
    """

    def __init__(self, embedding_dim, head_number, hidden_dim, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(embedding_dim, head_number)
        self.add_norm_1 = AddNorm(embedding_dim, dropout)

        self.feed_forward = FeedForward(embedding_dim, hidden_dim, dropout)
        self.add_norm_2 = AddNorm(embedding_dim, dropout)

    def forward(self, x):

        # Sub-layer 1 : self-attention + residual + norm
        attention_output, attention_weights = self.attention(x)
        x = self.add_norm_1(x, attention_output)

        # Sub-layer 2 : feed forward + residual + norm
        feed_forward_output = self.feed_forward(x)
        x = self.add_norm_2(x, feed_forward_output)

        return x, attention_weights


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    # Load dataset
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

    # Embedding
    embedding = nn.Embedding(
        num_embeddings=len(word_to_id),
        embedding_dim=EMBEDDING_DIM
    )

    # Positional Encoding
    positional_encoding = PositionalEncoding(EMBEDDING_DIM)

    # Encoder Block (attention + AddNorm + FFN + AddNorm)
    encoder_block = EncoderBlock(
        embedding_dim=EMBEDDING_DIM,
        head_number=HEAD_NUMBER,
        hidden_dim=FF_HIDDEN_DIM,
        dropout=DROPOUT
    )

    encoder_block.eval()  # deterministic: disables dropout

    for inputs, targets in dataloader:

        print("=" * 50)
        print("Token IDs")
        print("=" * 50)
        print(inputs)
        print()

        # Embeddings
        embedded_inputs = embedding(inputs)

        # Positional Encoding
        embedded_inputs = positional_encoding(embedded_inputs)

        # Encoder Block
        block_output, attention_weights = encoder_block(embedded_inputs)

        print("=" * 50)
        print("Input Shape")
        print("=" * 50)
        print(embedded_inputs.shape)
        print()

        print("=" * 50)
        print("Block Output Shape")
        print("=" * 50)
        print(block_output.shape)
        print()

        print("=" * 50)
        print("Attention Weights Shape")
        print("=" * 50)
        print(attention_weights.shape)
        print()

        # Check LayerNorm : mean ~ 0 and std ~ 1 on the last dimension
        print("=" * 50)
        print("LayerNorm Check (last dim)")
        print("=" * 50)
        print("mean :", block_output.mean(dim=-1))
        print("std  :", block_output.std(dim=-1))
        print()

        first_sentence = inputs[0]

        print("=" * 50)
        print("Tokens and Block Output")
        print("=" * 50)

        for i, token_id in enumerate(first_sentence):
            idx = token_id.item()
            word = id_to_word[idx]
            vector = block_output[0][i]

            print(f"{word:12} -> {idx:3d} -> {vector[:5]}")

        break


if __name__ == "__main__":
    main()