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

from lab14_transformer_encoder import TransformerEncoder
from lab17_decoder_block import DecoderBlock


# -------------------------------------------------
# Transformer Model
# -------------------------------------------------

class TransformerModel(nn.Module):
    """
    Complete Transformer Model

    Note

    TransformerEncoder (lab14) already contains
    its own embedding layer and positional encoding.
    The encoder therefore receives raw token ids.

    The decoder side keeps its own embedding and
    positional encoding, because DecoderBlock (lab17)
    does not contain them.

    Architecture

        Source Token Ids
              |
              v
    Transformer Encoder
    (embedding + position inside)
              |
              v
      Encoder Output
              |
              v
        Decoder Blocks
              ^
              |
      Target Embedding
              |
    Positional Encoding
              |
              v
        Decoder Output
              |
              v
        Linear Projection
              |
              v
      Vocabulary Logits
    """

    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        embedding_dim,
        head_number,
        hidden_dim,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.1,
    ):

        super().__init__()

        # -----------------------------------------
        # Transformer Encoder
        # -----------------------------------------

        self.encoder = TransformerEncoder(
            vocab_size=source_vocab_size,
            embedding_dim=embedding_dim,
            head_number=head_number,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            dropout=dropout
        )

        # -----------------------------------------
        # Target Embedding
        # -----------------------------------------

        self.target_embedding = nn.Embedding(
            target_vocab_size,
            embedding_dim
        )

        self.position = PositionalEncoding(
            embedding_dim
        )

        # -----------------------------------------
        # Transformer Decoder
        # -----------------------------------------

        self.decoder_layers = nn.ModuleList(

            [
                DecoderBlock(
                    embedding_dim=embedding_dim,
                    head_number=head_number,
                    hidden_dim=hidden_dim,
                    dropout=dropout
                )

                for _ in range(num_decoder_layers)
            ]

        )

        # -----------------------------------------
        # Output Layer
        # -----------------------------------------

        self.output_layer = nn.Linear(
            embedding_dim,
            target_vocab_size
        )

    # ---------------------------------------------
    # Forward
    # ---------------------------------------------

    def forward(
        self,
        source,
        target
    ):

        # -----------------------------------------
        # Encoder
        #
        # source : raw token ids
        # (batch, source_seq)
        # -----------------------------------------

        encoder_output, encoder_attention = self.encoder(
            source
        )

        # -----------------------------------------
        # Target Embedding
        # -----------------------------------------

        target = self.target_embedding(target)
        target = self.position(target)

        # -----------------------------------------
        # Decoder
        # -----------------------------------------

        decoder_output = target

        masked_attention = None
        cross_attention = None

        for decoder in self.decoder_layers:

            decoder_output, masked_attention, cross_attention = decoder(
                decoder_output,
                encoder_output
            )

        # -----------------------------------------
        # Output Projection
        # -----------------------------------------

        logits = self.output_layer(
            decoder_output
        )

        return (
            logits,
            encoder_attention,
            masked_attention,
            cross_attention
        )


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATASET_PATH_EN = Path("datasets/sample.txt")
DATASET_PATH_FR = Path("datasets/sample_fr.txt")

SEQUENCE_LENGTH = 5
BATCH_SIZE = 2

EMBEDDING_DIM = 128
HEAD_NUMBER = 4
HIDDEN_DIM = EMBEDDING_DIM * 4

NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

DROPOUT = 0.1


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("LAB 18 - COMPLETE TRANSFORMER")
    print("=" * 60)

    # -------------------------------------------------
    # Load datasets
    # -------------------------------------------------

    text_en = load_text(DATASET_PATH_EN)
    text_fr = load_text(DATASET_PATH_FR)

    text_en = clean_text(text_en)
    text_fr = clean_text(text_fr)

    # -------------------------------------------------
    # Build vocabularies
    # -------------------------------------------------

    word_to_id_en, id_to_word_en = build_vocabulary(
        text_en
    )

    word_to_id_fr, id_to_word_fr = build_vocabulary(
        text_fr
    )

    # -------------------------------------------------
    # Encode datasets
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
    # Create datasets
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
    # DataLoaders
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
    # Create Transformer
    # -------------------------------------------------

    model = TransformerModel(
        source_vocab_size=len(word_to_id_en),
        target_vocab_size=len(word_to_id_fr),
        embedding_dim=EMBEDDING_DIM,
        head_number=HEAD_NUMBER,
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT
    )

    model.eval()

    print()

    print("=" * 60)
    print("MODEL CREATED")
    print("=" * 60)

    print(model)

    print()

    # -------------------------------------------------
    # Iterate over one batch
    # -------------------------------------------------

    for batch_index, (
        (inputs_en, targets_en),
        (inputs_fr, targets_fr)
    ) in enumerate(
        zip(
            dataloader_en,
            dataloader_fr
        )
    ):

        print("=" * 60)
        print(f"BATCH {batch_index}")
        print("=" * 60)

        print("English Input :", inputs_en.shape)
        print("French Input  :", inputs_fr.shape)
        print()

        # -----------------------------------------
        # Complete Transformer Forward Pass
        # -----------------------------------------

        logits, encoder_attention, masked_attention, cross_attention = model(
            inputs_en,
            inputs_fr
        )

        print("=" * 60)
        print("TRANSFORMER OUTPUT")
        print("=" * 60)

        print("Logits Shape             :", logits.shape)
        print("Masked Attention Shape   :", masked_attention.shape)
        print("Cross Attention Shape    :", cross_attention.shape)
        print()

        # -----------------------------------------
        # Vocabulary Prediction
        # -----------------------------------------

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        print("=" * 60)
        print("PREDICTED TOKEN IDS")
        print("=" * 60)

        print(predictions)
        print()

        # -----------------------------------------
        # Predicted Words
        # -----------------------------------------

        print("=" * 60)
        print("PREDICTED WORDS")
        print("=" * 60)

        for sentence in predictions:

            words = []

            for token in sentence:

                token_id = token.item()

                if token_id in id_to_word_fr:
                    words.append(id_to_word_fr[token_id])
                else:
                    words.append("<UNK>")

            print(" ".join(words))

        print()

        # -----------------------------------------
        # Source Sentence
        # -----------------------------------------

        print("=" * 60)
        print("SOURCE SENTENCE")
        print("=" * 60)

        for sentence in inputs_en:

            words = []

            for token in sentence:

                token_id = token.item()

                words.append(
                    id_to_word_en[token_id]
                )

            print(" ".join(words))

        print()

        # -----------------------------------------
        # Ground Truth
        # -----------------------------------------

        print("=" * 60)
        print("TARGET SENTENCE")
        print("=" * 60)

        for sentence in inputs_fr:

            words = []

            for token in sentence:

                token_id = token.item()

                words.append(
                    id_to_word_fr[token_id]
                )

            print(" ".join(words))

        print()

        # -----------------------------------------
        # Decoder Masked Attention
        # -----------------------------------------

        print("=" * 60)
        print("MASKED ATTENTION (Head 0)")
        print("=" * 60)

        print(
            masked_attention[0, 0]
        )

        print()

        # -----------------------------------------
        # Cross Attention
        # -----------------------------------------

        print("=" * 60)
        print("CROSS ATTENTION (Head 0)")
        print("=" * 60)

        print(
            cross_attention[0, 0]
        )

        print()

        # -----------------------------------------
        # Logits of First Token
        # -----------------------------------------

        print("=" * 60)
        print("FIRST TOKEN LOGITS")
        print("=" * 60)

        print(
            logits[0, 0, :20]
        )

        print()

        # -----------------------------------------
        # Top-5 Predictions
        # -----------------------------------------

        values, indices = torch.topk(
            logits[0, 0],
            k=5
        )

        print("=" * 60)
        print("TOP 5 PREDICTIONS")
        print("=" * 60)

        for rank in range(5):

            token_id = indices[rank].item()

            word = id_to_word_fr[token_id]

            score = values[rank].item()

            print(
                f"{rank + 1}. {word:15} {score:.4f}"
            )

        print()

        break

    print("=" * 60)
    print("LAB 18 FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()