"""
==========================================================
Lab 19 - Transformer Training Loop
==========================================================

Objective:

1. Load source and target datasets.
2. Create the Transformer model.
3. Perform forward propagation.
4. Compute CrossEntropyLoss.
5. Perform backpropagation.
6. Update model parameters using Adam.
7. Train for multiple epochs.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lab06_positional_encoding import (
    TextDataset,
    build_vocabulary,
    clean_text,
    encode,
    load_text,
)

from lab18_transformer_model import TransformerModel


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATASET_PATH_EN = Path("datasets/sample.txt")
DATASET_PATH_FR = Path("datasets/sample_fr.txt")
MODEL_PATH = Path("models/transformer.pth")

SEQUENCE_LENGTH = 5
BATCH_SIZE = 2

EMBEDDING_DIM = 128
HEAD_NUMBER = 4
HIDDEN_DIM = EMBEDDING_DIM * 4

NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

DROPOUT = 0.1

LEARNING_RATE = 0.001
EPOCHS = 12


# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("LAB 19 - TRANSFORMER TRAINING")
    print("=" * 60)

    # -------------------------------------------------
    # Load text
    # -------------------------------------------------

    text_en = load_text(DATASET_PATH_EN)
    text_fr = load_text(DATASET_PATH_FR)

    text_en = clean_text(text_en)
    text_fr = clean_text(text_fr)

    # -------------------------------------------------
    # Vocabulary
    # -------------------------------------------------

    word_to_id_en, id_to_word_en = build_vocabulary(
        text_en
    )

    word_to_id_fr, id_to_word_fr = build_vocabulary(
        text_fr
    )

    print("English vocabulary:", len(word_to_id_en))
    print("French vocabulary :", len(word_to_id_fr))

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
    shuffle=False,
    drop_last=True
)

    dataloader_fr = DataLoader(
    dataset_fr,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)
    # -------------------------------------------------
    # Transformer
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

    model = model.to(device)

    # IMPORTANT:
    # Unlike Lab 18, we use train() here.
    model.train()

    # -------------------------------------------------
    # Loss
    # -------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("START TRAINING")
    print("=" * 60)

    for epoch in range(EPOCHS):

        total_loss = 0.0
        batch_count = 0

        for (
            (inputs_en, targets_en),
            (inputs_fr, targets_fr)
        ) in zip(dataloader_en, dataloader_fr):

            # -----------------------------------------
            # Move tensors to device
            # -----------------------------------------

            inputs_en = inputs_en.to(device)

            inputs_fr = inputs_fr.to(device)

            targets_fr = targets_fr.to(device)

            # -----------------------------------------
            # Reset gradients
            # -----------------------------------------

            optimizer.zero_grad()

            # -----------------------------------------
            # Forward Pass
            # -----------------------------------------

            logits, _, _, _ = model(
                inputs_en,
                inputs_fr
            )

            # logits shape:
            #
            # (batch,
            #  sequence_length,
            #  target_vocab_size)

            # -----------------------------------------
            # Prepare logits for CrossEntropyLoss
            # -----------------------------------------

            logits = logits.reshape(
                -1,
                logits.size(-1)
            )

            # becomes:
            #
            # (batch * sequence_length,
            #  target_vocab_size)

            # -----------------------------------------
            # Prepare targets
            # -----------------------------------------

            targets = targets_fr.reshape(-1)

            # becomes:
            #
            # (batch * sequence_length)

            # -----------------------------------------
            # Loss
            # -----------------------------------------

            loss = criterion(
                logits,
                targets
            )

            # -----------------------------------------
            # Backpropagation
            # -----------------------------------------

            loss.backward()

            # -----------------------------------------
            # Update Parameters
            # -----------------------------------------

            optimizer.step()

            # -----------------------------------------
            # Statistics
            # -----------------------------------------

            total_loss += loss.item()

            batch_count += 1

        # ---------------------------------------------
        # Average Loss
        # ---------------------------------------------

        average_loss = total_loss / batch_count

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.4f}"
        )

    print()
    print("=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)
    # -------------------------------------------------
    # Save Model
    # -------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            "source_vocab_size": len(word_to_id_en),
            "target_vocab_size": len(word_to_id_fr),

            "embedding_dim": EMBEDDING_DIM,
            "head_number": HEAD_NUMBER,
            "hidden_dim": HIDDEN_DIM,

            "num_encoder_layers": NUM_ENCODER_LAYERS,
            "num_decoder_layers": NUM_DECODER_LAYERS,

            "word_to_id_en": word_to_id_en,
            "id_to_word_en": id_to_word_en,

            "word_to_id_fr": word_to_id_fr,
            "id_to_word_fr": id_to_word_fr,
        },
        MODEL_PATH
    )

    print()
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()