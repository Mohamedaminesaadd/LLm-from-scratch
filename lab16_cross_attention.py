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

from lab15_masked_multi_head_attention import (
    MaskedMultiHeadSelfAttention,
)


# -------------------------------------------------
# Cross Attention (Single Head)
# -------------------------------------------------

class CrossAttention(nn.Module):
    """
    Single-Head Cross Attention

    Q comes from the decoder.
    K and V come from the encoder.
    """

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, decoder_input, encoder_output):

        # -----------------------------------------
        # Linear projections
        # -----------------------------------------

        Q = self.query(decoder_input)
        K = self.key(encoder_output)
        V = self.value(encoder_output)

        # -----------------------------------------
        # Attention scores
        # -----------------------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.embedding_dim)

        # -----------------------------------------
        # Softmax
        # -----------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # -----------------------------------------
        # Weighted sum
        # -----------------------------------------

        output = torch.matmul(
            attention_weights,
            V
        )

        return output, attention_weights


# -------------------------------------------------
# Multi-Head Cross Attention
# -------------------------------------------------

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross Attention

    Q : Decoder

    K,V : Encoder
    """

    def __init__(self,
                 embedding_dim,
                 head_number):

        super().__init__()

        assert embedding_dim % head_number == 0

        self.embedding_dim = embedding_dim
        self.head_number = head_number

        self.head_dim = embedding_dim // head_number

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self,
                decoder_input,
                encoder_output):

        batch_size = decoder_input.size(0)

        # -----------------------------------------
        # Linear projections
        # -----------------------------------------

        Q = self.query(decoder_input)
        K = self.key(encoder_output)
        V = self.value(encoder_output)

        # -----------------------------------------
        # Split into heads
        # -----------------------------------------

        Q = Q.view(
            batch_size,
            -1,
            self.head_number,
            self.head_dim
        )

        K = K.view(
            batch_size,
            -1,
            self.head_number,
            self.head_dim
        )

        V = V.view(
            batch_size,
            -1,
            self.head_number,
            self.head_dim
        )

        # -----------------------------------------
        # Move heads dimension
        # -----------------------------------------

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Shapes
        #
        # Q
        # (batch, heads, target_seq, head_dim)
        #
        # K
        # (batch, heads, source_seq, head_dim)
        #
        # V
        # (batch, heads, source_seq, head_dim)

        # -----------------------------------------
        # Attention scores
        # -----------------------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        # Shape
        #
        # (batch, heads,
        #  target_seq,
        #  source_seq)

        # -----------------------------------------
        # Softmax
        # -----------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # -----------------------------------------
        # Weighted sum
        # -----------------------------------------

        attention = torch.matmul(
            attention_weights,
            V
        )

        # -----------------------------------------
        # Concatenate heads
        # -----------------------------------------

        attention = attention.transpose(1, 2)

        attention = attention.contiguous().view(
            batch_size,
            -1,
            self.embedding_dim
        )

        # -----------------------------------------
        # Final Linear Layer
        # -----------------------------------------

        output = self.output(attention)

        return output, attention_weights


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
    print("LAB 16 - CROSS ATTENTION")
    print("=" * 60)

    # -------------------------------------------------
    # Load datasets
    # -------------------------------------------------

    text_en = load_text(DATASET_PATH_EN)
    text_fr = load_text(DATASET_PATH_FR)

    text_en = clean_text(text_en)
    text_fr = clean_text(text_fr)

    print("English dataset length :", len(text_en))
    print("French dataset length  :", len(text_fr))
    print()

    # -------------------------------------------------
    # Build vocabularies
    # -------------------------------------------------

    word_to_id_en, id_to_word_en = build_vocabulary(text_en)
    word_to_id_fr, id_to_word_fr = build_vocabulary(text_fr)

    print("English vocabulary :", len(word_to_id_en))
    print("French vocabulary  :", len(word_to_id_fr))
    print()

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
    # Embeddings
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
    # Decoder Masked Self Attention
    # -------------------------------------------------

    decoder_attention = MaskedMultiHeadSelfAttention(
        embedding_dim=EMBEDDING_DIM,
        head_number=HEAD_NUMBER
    )

    decoder_attention.eval()

    # -------------------------------------------------
    # Cross Attention
    # -------------------------------------------------

    cross_attention = MultiHeadCrossAttention(
        embedding_dim=EMBEDDING_DIM,
        head_number=HEAD_NUMBER
    )

    cross_attention.eval()

    print("=" * 60)
    print("MODEL CREATED")
    print("=" * 60)
    print()

    # -------------------------------------------------
    # Iterate over both datasets
    # -------------------------------------------------

    for batch_index, (
        (inputs_en, targets_en),
        (inputs_fr, targets_fr)
    ) in enumerate(zip(dataloader_en, dataloader_fr)):

        print("=" * 60)
        print(f"BATCH {batch_index}")
        print("=" * 60)

        print("English Input Shape :", inputs_en.shape)
        print("French  Input Shape :", inputs_fr.shape)
        print()

        # -----------------------------------------
        # Encoder embeddings
        # -----------------------------------------

        encoder_embeddings = embedding_en(inputs_en)

        encoder_embeddings = positional_encoding(
            encoder_embeddings
        )

        # -----------------------------------------
        # Decoder embeddings
        # -----------------------------------------

        decoder_embeddings = embedding_fr(inputs_fr)

        decoder_embeddings = positional_encoding(
            decoder_embeddings
        )

        print("Encoder Embeddings :", encoder_embeddings.shape)
        print("Decoder Embeddings :", decoder_embeddings.shape)
        print()

        # -----------------------------------------
        # Encoder
        # -----------------------------------------

        encoder_output, encoder_attention = encoder(
            encoder_embeddings
        )

        print("=" * 60)
        print("ENCODER")
        print("=" * 60)

        print("Encoder Output Shape :", encoder_output.shape)
        print("Encoder Attention Shape :", encoder_attention.shape)
        print()

        # -----------------------------------------
        # Decoder (Masked Self Attention)
        # -----------------------------------------

        decoder_output, decoder_attention_weights = decoder_attention(
            decoder_embeddings
        )

        print("=" * 60)
        print("MASKED SELF ATTENTION")
        print("=" * 60)

        print("Decoder Output Shape :", decoder_output.shape)
        print("Decoder Attention Shape :", decoder_attention_weights.shape)
        print()

        # -----------------------------------------
        # Cross Attention
        # -----------------------------------------

        cross_output, cross_attention_weights = cross_attention(
            decoder_output,
            encoder_output
        )

        print("=" * 60)
        print("CROSS ATTENTION")
        print("=" * 60)

        print("Cross Output Shape :", cross_output.shape)
        print("Cross Attention Shape :", cross_attention_weights.shape)
        print()

        # -----------------------------------------
        # Display Attention Matrix
        # -----------------------------------------

        print("=" * 60)
        print("HEAD 0 ATTENTION MATRIX")
        print("=" * 60)

        print(cross_attention_weights[0, 0])
        print()

        # -----------------------------------------
        # Display Token Representations
        # -----------------------------------------

        print("=" * 60)
        print("DECODER TOKEN REPRESENTATIONS")
        print("=" * 60)

        for i in range(inputs_fr.shape[1]):

            token_id = inputs_fr[0, i].item()

            token = id_to_word_fr[token_id]

            print(
                f"{token:12} -> {cross_output[0, i, :8]}"
            )

        break


if __name__ == "__main__":
    main()