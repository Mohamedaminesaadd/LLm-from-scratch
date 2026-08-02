"""
==========================================================
Lab 20 - Transformer Validation
==========================================================

Objective:

1. Load the trained Transformer checkpoint.
2. Recreate the Transformer architecture.
3. Load the trained weights.
4. Switch the model to evaluation mode.
5. Disable gradient computation.
6. Perform forward propagation.
7. Compute validation loss.
8. Compute token accuracy.
9. Display example predictions.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lab06_positional_encoding import (
    TextDataset,
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
    print("LAB 20 - TRANSFORMER VALIDATION")
    print("=" * 60)

    # -------------------------------------------------
    # Check checkpoint
    # -------------------------------------------------

    if not MODEL_PATH.exists():

        print("ERROR: Model checkpoint not found.")
        print("Expected:", MODEL_PATH)
        print("Run Lab 19 first.")

        return

    # -------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    print("Checkpoint loaded successfully.")

    # -------------------------------------------------
    # Recover configuration
    # -------------------------------------------------

    source_vocab_size = checkpoint[
        "source_vocab_size"
    ]

    target_vocab_size = checkpoint[
        "target_vocab_size"
    ]

    embedding_dim = checkpoint[
        "embedding_dim"
    ]

    head_number = checkpoint[
        "head_number"
    ]

    hidden_dim = checkpoint[
        "hidden_dim"
    ]

    num_encoder_layers = checkpoint[
        "num_encoder_layers"
    ]

    num_decoder_layers = checkpoint[
        "num_decoder_layers"
    ]

    # -------------------------------------------------
    # Recover vocabularies
    # -------------------------------------------------

    word_to_id_en = checkpoint[
        "word_to_id_en"
    ]

    id_to_word_en = checkpoint[
        "id_to_word_en"
    ]

    word_to_id_fr = checkpoint[
        "word_to_id_fr"
    ]

    id_to_word_fr = checkpoint[
        "id_to_word_fr"
    ]

    print()
    print("Source vocabulary size :", source_vocab_size)
    print("Target vocabulary size :", target_vocab_size)
    print("Embedding dimension    :", embedding_dim)
    print("Number of heads        :", head_number)

    # -------------------------------------------------
    # Recreate Transformer
    # -------------------------------------------------

    model = TransformerModel(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        embedding_dim=embedding_dim,
        head_number=head_number,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=0.1
    )

    # -------------------------------------------------
    # Load trained parameters
    # -------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    # -------------------------------------------------
    # Evaluation Mode
    # -------------------------------------------------

    model.eval()

    print()
    print("=" * 60)
    print("TRAINED MODEL READY")
    print("=" * 60)

    # -------------------------------------------------
    # Load text
    # -------------------------------------------------

    text_en = load_text(
        DATASET_PATH_EN
    )

    text_fr = load_text(
        DATASET_PATH_FR
    )

    text_en = clean_text(text_en)
    text_fr = clean_text(text_fr)

    # -------------------------------------------------
    # Encode using TRAINING vocabulary
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
    # Loss Function
    # -------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    total_loss = 0.0

    total_correct = 0

    total_tokens = 0

    batch_count = 0

    # We will keep one example for display
    example_source = None
    example_decoder_input = None
    example_target = None
    example_prediction = None

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("START VALIDATION")
    print("=" * 60)

    # -------------------------------------------------
    # Disable Gradient Calculation
    # -------------------------------------------------

    with torch.no_grad():

        for (
            (inputs_en, targets_en),
            (inputs_fr, targets_fr)
        ) in zip(
            dataloader_en,
            dataloader_fr
        ):

            # -----------------------------------------
            # Move to device
            # -----------------------------------------

            inputs_en = inputs_en.to(device)

            inputs_fr = inputs_fr.to(device)

            targets_fr = targets_fr.to(device)

            # -----------------------------------------
            # Forward Pass
            # -----------------------------------------

            logits, _, _, _ = model(
                inputs_en,
                inputs_fr
            )

            # logits shape:
            #
            # [batch_size,
            #  sequence_length,
            #  target_vocab_size]

            # -----------------------------------------
            # Compute Loss
            # -----------------------------------------

            loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets_fr.reshape(-1)
            )

            total_loss += loss.item()

            batch_count += 1

            # -----------------------------------------
            # Predictions
            # -----------------------------------------

            predictions = torch.argmax(
                logits,
                dim=-1
            )

            # predictions:
            #
            # [batch_size, sequence_length]

            # -----------------------------------------
            # Token Accuracy
            # -----------------------------------------

            correct = (
                predictions == targets_fr
            ).sum().item()

            total_correct += correct

            total_tokens += targets_fr.numel()

            # -----------------------------------------
            # Save first example
            # -----------------------------------------

            if example_source is None:

                example_source = (
                    inputs_en[0]
                    .detach()
                    .cpu()
                )

                example_decoder_input = (
                    inputs_fr[0]
                    .detach()
                    .cpu()
                )

                example_target = (
                    targets_fr[0]
                    .detach()
                    .cpu()
                )

                example_prediction = (
                    predictions[0]
                    .detach()
                    .cpu()
                )

    # -------------------------------------------------
    # Avoid division by zero
    # -------------------------------------------------

    if batch_count == 0:

        print("ERROR: No validation batches.")
        return

    # -------------------------------------------------
    # Average Validation Loss
    # -------------------------------------------------

    validation_loss = (
        total_loss / batch_count
    )

    # -------------------------------------------------
    # Accuracy
    # -------------------------------------------------

    accuracy = (
        total_correct / total_tokens
    ) * 100

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"Validation Loss : "
        f"{validation_loss:.4f}"
    )

    print(
        f"Token Accuracy  : "
        f"{accuracy:.2f}%"
    )

    print(
        f"Correct Tokens  : "
        f"{total_correct}/{total_tokens}"
    )

    # -------------------------------------------------
    # Display Example
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("EXAMPLE")
    print("=" * 60)

    # -------------------------------------------------
    # Source
    # -------------------------------------------------

    source_words = []

    for token in example_source:

        token_id = token.item()

        word = id_to_word_en.get(
            token_id,
            "<UNK>"
        )

        source_words.append(word)

    print()
    print("SOURCE ENGLISH:")

    print(
        " ".join(source_words)
    )

    # -------------------------------------------------
    # Decoder Input
    # -------------------------------------------------

    decoder_words = []

    for token in example_decoder_input:

        token_id = token.item()

        word = id_to_word_fr.get(
            token_id,
            "<UNK>"
        )

        decoder_words.append(word)

    print()
    print("DECODER INPUT:")

    print(
        " ".join(decoder_words)
    )

    # -------------------------------------------------
    # Expected Target
    # -------------------------------------------------

    target_words = []

    for token in example_target:

        token_id = token.item()

        word = id_to_word_fr.get(
            token_id,
            "<UNK>"
        )

        target_words.append(word)

    print()
    print("EXPECTED:")

    print(
        " ".join(target_words)
    )

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    prediction_words = []

    for token in example_prediction:

        token_id = token.item()

        word = id_to_word_fr.get(
            token_id,
            "<UNK>"
        )

        prediction_words.append(word)

    print()
    print("PREDICTED:")

    print(
        " ".join(prediction_words)
    )

    # -------------------------------------------------
    # Compare Token by Token
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("TOKEN COMPARISON")
    print("=" * 60)

    for position, (
        expected_token,
        predicted_token
    ) in enumerate(
        zip(
            example_target,
            example_prediction
        )
    ):

        expected_id = expected_token.item()

        predicted_id = predicted_token.item()

        expected_word = id_to_word_fr.get(
            expected_id,
            "<UNK>"
        )

        predicted_word = id_to_word_fr.get(
            predicted_id,
            "<UNK>"
        )

        correct = (
            expected_id == predicted_id
        )

        status = (
            "OK"
            if correct
            else "WRONG"
        )

        print(
            f"Position {position}: "
            f"expected={expected_word:15} "
            f"predicted={predicted_word:15} "
            f"{status}"
        )

    print()
    print("=" * 60)
    print("VALIDATION FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()