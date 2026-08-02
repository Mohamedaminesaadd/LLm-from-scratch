import torch
import torch.nn as nn

from lab13_encoder_block import (
    AddNorm,
    FeedForward,
)

from lab15_masked_multi_head_attention import (
    MaskedMultiHeadSelfAttention,
)

from lab16_cross_attention import (
    MultiHeadCrossAttention,
)


# -------------------------------------------------
# Decoder Block
# -------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block

    Architecture

            decoder_input
                  │
                  ▼
    Masked Multi-Head Self Attention
                  │
                  ▼
            Add + LayerNorm
                  │
                  ▼
        Multi-Head Cross Attention
                  │
                  ▼
            Add + LayerNorm
                  │
                  ▼
          Feed Forward Network
                  │
                  ▼
            Add + LayerNorm
                  │
                  ▼
            decoder_output
    """

    def __init__(
        self,
        embedding_dim,
        head_number,
        hidden_dim,
        dropout=0.1
    ):

        super().__init__()

        # ---------------------------------------
        # First Attention
        # ---------------------------------------

        self.masked_attention = MaskedMultiHeadSelfAttention(
            embedding_dim,
            head_number
        )

        self.add_norm1 = AddNorm(
            embedding_dim,
            dropout
        )

        # ---------------------------------------
        # Cross Attention
        # ---------------------------------------

        self.cross_attention = MultiHeadCrossAttention(
            embedding_dim,
            head_number
        )

        self.add_norm2 = AddNorm(
            embedding_dim,
            dropout
        )

        # ---------------------------------------
        # Feed Forward
        # ---------------------------------------

        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim,
            dropout
        )

        self.add_norm3 = AddNorm(
            embedding_dim,
            dropout
        )

    def forward(
        self,
        decoder_input,
        encoder_output
    ):

        # ---------------------------------------
        # Step 1
        # Masked Multi Head Attention
        # ---------------------------------------

        masked_output, masked_weights = self.masked_attention(
            decoder_input
        )

        x = self.add_norm1(
            decoder_input,
            masked_output
        )

        # ---------------------------------------
        # Step 2
        # Cross Attention
        # ---------------------------------------

        cross_output, cross_weights = self.cross_attention(
            x,
            encoder_output
        )

        x = self.add_norm2(
            x,
            cross_output
        )

        # ---------------------------------------
        # Step 3
        # Feed Forward
        # ---------------------------------------

        ff_output = self.feed_forward(
            x
        )

        output = self.add_norm3(
            x,
            ff_output
        )

        return (
            output,
            masked_weights,
            cross_weights
        )


import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lab06_positional_encoding import (
    PositionalEncoding,
    TextDataset,
    build_vocabulary,
    clean_text,
    encode,
    load_text,
)

from lab13_encoder_block import (
    TransformerBlock,
)

# DecoderBlock is implemented in Part 1
# from this file

# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATASET_PATH_EN = Path("datasets/sample.txt")
DATASET_PATH_FR = Path("datasets/sample_fr.txt")

SEQUENCE_LENGTH = 5
BATCH_SIZE = 2

EMBEDDING_DIM = 128
HEAD_NUMBER = 4
FF_HIDDEN_DIM = EMBEDDING_DIM * 4
DROPOUT = 0.1


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("LAB 17 - DECODER BLOCK")
    print("=" * 60)

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------

    text_en = load_text(DATASET_PATH_EN)
    text_fr = load_text(DATASET_PATH_FR)

    text_en = clean_text(text_en)
    text_fr = clean_text(text_fr)

    # -------------------------------------------------
    # Vocabulary
    # -------------------------------------------------

    word_to_id_en, id_to_word_en = build_vocabulary(text_en)
    word_to_id_fr, id_to_word_fr = build_vocabulary(text_fr)

    # -------------------------------------------------
    # Encode
    # -------------------------------------------------

    token_ids_en = encode(
        text_en,
        word_to_id_en
    )

    token_ids_fr = encode(
        text_fr,
        word_to_id_fr
    )

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    dataset_en = TextDataset(
        token_ids_en,
        SEQUENCE_LENGTH
    )

    dataset_fr = TextDataset(
        token_ids_fr,
        SEQUENCE_LENGTH
    )

    # -------------------------------------------------
    # DataLoader
    # -------------------------------------------------

    dataloader_en = DataLoader(
        dataset_en,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    dataloader_fr = DataLoader(
        dataset_fr,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -------------------------------------------------
    # Embedding
    # -------------------------------------------------

    embedding_en = nn.Embedding(
        len(word_to_id_en),
        EMBEDDING_DIM
    )

    embedding_fr = nn.Embedding(
        len(word_to_id_fr),
        EMBEDDING_DIM
    )

    positional_encoding = PositionalEncoding(
        EMBEDDING_DIM
    )

    # -------------------------------------------------
    # Encoder
    # -------------------------------------------------

    encoder = TransformerBlock(
        embedding_dim=EMBEDDING_DIM,
        head_number=HEAD_NUMBER,
        hidden_dim=FF_HIDDEN_DIM,
        dropout=DROPOUT
    )

    encoder.eval()

    # -------------------------------------------------
    # Decoder Block
    # -------------------------------------------------

    decoder = DecoderBlock(
        embedding_dim=EMBEDDING_DIM,
        head_number=HEAD_NUMBER,
        hidden_dim=FF_HIDDEN_DIM,
        dropout=DROPOUT
    )

    decoder.eval()

    print("=" * 60)
    print("MODEL CREATED")
    print("=" * 60)

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------

    for batch_index, (
        (inputs_en, targets_en),
        (inputs_fr, targets_fr)
    ) in enumerate(zip(dataloader_en, dataloader_fr)):

        print("=" * 60)
        print(f"BATCH {batch_index}")
        print("=" * 60)

        # -----------------------------
        # Encoder Embedding
        # -----------------------------

        encoder_embeddings = embedding_en(inputs_en)
        encoder_embeddings = positional_encoding(
            encoder_embeddings
        )

        # -----------------------------
        # Decoder Embedding
        # -----------------------------

        decoder_embeddings = embedding_fr(inputs_fr)
        decoder_embeddings = positional_encoding(
            decoder_embeddings
        )

        print("Encoder Embedding :", encoder_embeddings.shape)
        print("Decoder Embedding :", decoder_embeddings.shape)
        print()

        # Part 3 starts here...
                # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        encoder_output, encoder_attention = encoder(
            encoder_embeddings
        )

        print("=" * 60)
        print("ENCODER")
        print("=" * 60)

        print("Encoder Output Shape :", encoder_output.shape)
        print("Encoder Attention Shape :", encoder_attention.shape)
        print()

        # -------------------------------------------------
        # Decoder Block
        # -------------------------------------------------

        decoder_output, masked_weights, cross_weights = decoder(
            decoder_embeddings,
            encoder_output
        )

        print("=" * 60)
        print("DECODER BLOCK")
        print("=" * 60)

        print("Decoder Output Shape :", decoder_output.shape)
        print()

        print("Masked Attention Shape :", masked_weights.shape)
        print("Cross Attention Shape :", cross_weights.shape)
        print()

        # -------------------------------------------------
        # Masked Attention
        # -------------------------------------------------

        print("=" * 60)
        print("MASKED SELF ATTENTION")
        print("=" * 60)

        print(masked_weights[0, 0])
        print()

        # -------------------------------------------------
        # Cross Attention
        # -------------------------------------------------

        print("=" * 60)
        print("CROSS ATTENTION")
        print("=" * 60)

        print(cross_weights[0, 0])
        print()

        # -------------------------------------------------
        # Decoder Token Representations
        # -------------------------------------------------

        print("=" * 60)
        print("DECODER TOKEN REPRESENTATIONS")
        print("=" * 60)

        first_sentence = inputs_fr[0]

        for i, token in enumerate(first_sentence):

            token_id = token.item()

            word = id_to_word_fr[token_id]

            print(
                f"{word:15} -> {decoder_output[0, i, :8]}"
            )

        print()

        # -------------------------------------------------
        # Encoder Token Representations
        # -------------------------------------------------

        print("=" * 60)
        print("ENCODER TOKEN REPRESENTATIONS")
        print("=" * 60)

        first_sentence = inputs_en[0]

        for i, token in enumerate(first_sentence):

            token_id = token.item()

            word = id_to_word_en[token_id]

            print(
                f"{word:15} -> {encoder_output[0, i, :8]}"
            )

        print()

        break

    print("=" * 60)
    print("LAB 17 FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()