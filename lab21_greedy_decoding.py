"""
==========================================================
Lab 21 - Greedy Decoding
==========================================================

Objective:

1. Load the trained Transformer checkpoint.
2. Recreate the Transformer architecture.
3. Load the trained weights.
4. Encode a source sentence.
5. Start decoder generation with <BOS>.
6. Predict one token at a time.
7. Always select the token with the highest logit.
8. Stop when <EOS> is generated or max_length is reached.
9. Decode generated token IDs back to text.
"""

from pathlib import Path

import torch

from lab06_positional_encoding import (
    clean_text,
    encode,
)

from lab18_transformer_model import TransformerModel


# -------------------------------------------------
# Configuration
# -------------------------------------------------

MODEL_PATH = Path("models/transformer.pth")

MAX_GENERATION_LENGTH = 20


# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -------------------------------------------------
# Load Model
# -------------------------------------------------

def load_model():

    print("=" * 60)
    print("LOADING TRANSFORMER")
    print("=" * 60)

    # -------------------------------------------------
    # Check checkpoint
    # -------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            f"Run Lab 19 first."
        )

    # -------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

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

    # -------------------------------------------------
    # Recreate model
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
    # Load learned weights
    # -------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    # -------------------------------------------------
    # Evaluation mode
    # -------------------------------------------------

    model.eval()

    print("Model loaded successfully.")
    print()

    print(
        "Source vocabulary size:",
        source_vocab_size
    )

    print(
        "Target vocabulary size:",
        target_vocab_size
    )

    print()

    return (
        model,
        word_to_id_en,
        id_to_word_en,
        word_to_id_fr,
        id_to_word_fr
    )


# -------------------------------------------------
# Encode Source Sentence
# -------------------------------------------------

def encode_source(
    sentence,
    word_to_id_en
):

    # -------------------------------------------------
    # Clean text
    # -------------------------------------------------

    sentence = clean_text(
        sentence
    )

    # -------------------------------------------------
    # Encode
    # -------------------------------------------------

    token_ids = encode(
        sentence,
        word_to_id_en
    )

    if len(token_ids) == 0:

        raise ValueError(
            "The source sentence produced no tokens."
        )

    # -------------------------------------------------
    # Convert to tensor
    # -------------------------------------------------

    source = torch.tensor(
        token_ids,
        dtype=torch.long
    )

    # Before:
    #
    # [seq_length]
    #
    # After unsqueeze:
    #
    # [1, seq_length]

    source = source.unsqueeze(0)

    source = source.to(device)

    return source


# -------------------------------------------------
# Greedy Decoding
# -------------------------------------------------

def greedy_decode(
    model,
    source,
    word_to_id_fr,
    max_length=20
):

    # -------------------------------------------------
    # Get BOS and EOS
    # -------------------------------------------------

    if "<BOS>" not in word_to_id_fr:

        raise ValueError(
            "<BOS> token does not exist in the "
            "French vocabulary.\n"
            "Your training vocabulary must contain "
            "<BOS> for autoregressive decoding."
        )

    if "<EOS>" not in word_to_id_fr:

        raise ValueError(
            "<EOS> token does not exist in the "
            "French vocabulary.\n"
            "Your training vocabulary must contain "
            "<EOS> for autoregressive decoding."
        )

    bos_id = word_to_id_fr[
        "<BOS>"
    ]

    eos_id = word_to_id_fr[
        "<EOS>"
    ]

    # -------------------------------------------------
    # Start generation
    # -------------------------------------------------

    generated = torch.tensor(
        [[bos_id]],
        dtype=torch.long,
        device=device
    )

    print("=" * 60)
    print("GREEDY DECODING")
    print("=" * 60)

    print(
        "Initial decoder input:",
        generated
    )

    print()

    # -------------------------------------------------
    # No gradients during generation
    # -------------------------------------------------

    with torch.no_grad():

        for step in range(max_length):

            # -----------------------------------------
            # Forward Pass
            # -----------------------------------------

            logits, _, _, _ = model(
                source,
                generated
            )

            # logits shape:
            #
            # [batch,
            #  decoder_sequence_length,
            #  target_vocabulary_size]

            # -----------------------------------------
            # Last decoder position
            # -----------------------------------------

            last_token_logits = logits[
                :,
                -1,
                :
            ]

            # Shape:
            #
            # [1, target_vocab_size]

            # -----------------------------------------
            # Greedy selection
            # -----------------------------------------

            next_token = torch.argmax(
                last_token_logits,
                dim=-1
            )

            # Shape:
            #
            # [1]

            # -----------------------------------------
            # Add sequence dimension
            # -----------------------------------------

            next_token = next_token.unsqueeze(
                1
            )

            # Shape:
            #
            # [1, 1]

            # -----------------------------------------
            # Append token
            # -----------------------------------------

            generated = torch.cat(
                [
                    generated,
                    next_token
                ],
                dim=1
            )

            token_id = next_token.item()

            print(
                f"Step {step + 1:2d} "
                f"-> token ID: {token_id}"
            )

            # -----------------------------------------
            # Stop if EOS
            # -----------------------------------------

            if token_id == eos_id:

                print()
                print("<EOS> generated.")
                print("Stopping generation.")

                break

    return generated


# -------------------------------------------------
# Decode Generated Tokens
# -------------------------------------------------

def decode_generated(
    generated,
    id_to_word_fr
):

    token_ids = generated[
        0
    ].tolist()

    words = []

    for token_id in token_ids:

        word = id_to_word_fr.get(
            token_id,
            "<UNK>"
        )

        # Do not display BOS
        if word == "<BOS>":
            continue

        # Stop at EOS
        if word == "<EOS>":
            break

        # Optional:
        # do not display padding
        if word == "<PAD>":
            continue

        words.append(word)

    return " ".join(words)


# -------------------------------------------------
# Display Generated Tokens
# -------------------------------------------------

def display_tokens(
    generated,
    id_to_word_fr
):

    print()
    print("=" * 60)
    print("GENERATED TOKENS")
    print("=" * 60)

    for position, token in enumerate(
        generated[0]
    ):

        token_id = token.item()

        word = id_to_word_fr.get(
            token_id,
            "<UNK>"
        )

        print(
            f"{position:2d} "
            f"| ID = {token_id:4d} "
            f"| {word}"
        )


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("LAB 21 - GREEDY DECODING")
    print("=" * 60)

    print()

    # -------------------------------------------------
    # Load trained Transformer
    # -------------------------------------------------

    (
        model,
        word_to_id_en,
        id_to_word_en,
        word_to_id_fr,
        id_to_word_fr

    ) = load_model()

    # -------------------------------------------------
    # Check special tokens
    # -------------------------------------------------

    print("=" * 60)
    print("SPECIAL TOKENS")
    print("=" * 60)

    for token in [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>"
    ]:

        print(
            f"{token:8} -> "
            f"{word_to_id_fr.get(token, 'NOT FOUND')}"
        )

    print()

    # -------------------------------------------------
    # Source sentence
    # -------------------------------------------------

    sentence = input(
        "Enter an English sentence: "
    )

    print()

    print("=" * 60)
    print("SOURCE SENTENCE")
    print("=" * 60)

    print(sentence)

    # -------------------------------------------------
    # Encode source
    # -------------------------------------------------

    source = encode_source(
        sentence,
        word_to_id_en
    )

    print()
    print("Source IDs:")

    print(source)

    print()

    print(
        "Source shape:",
        source.shape
    )

    # -------------------------------------------------
    # Display source tokens
    # -------------------------------------------------

    print()

    print("=" * 60)
    print("SOURCE TOKENS")
    print("=" * 60)

    for token in source[0]:

        token_id = token.item()

        word = id_to_word_en.get(
            token_id,
            "<UNK>"
        )

        print(
            f"{token_id:4d} -> {word}"
        )

    # -------------------------------------------------
    # Greedy decoding
    # -------------------------------------------------

    generated = greedy_decode(
        model=model,
        source=source,
        word_to_id_fr=word_to_id_fr,
        max_length=MAX_GENERATION_LENGTH
    )

    # -------------------------------------------------
    # Display generated IDs
    # -------------------------------------------------

    print()

    print("=" * 60)
    print("GENERATED TOKEN IDS")
    print("=" * 60)

    print(generated)

    # -------------------------------------------------
    # Display generated tokens
    # -------------------------------------------------

    display_tokens(
        generated,
        id_to_word_fr
    )

    # -------------------------------------------------
    # Convert IDs -> text
    # -------------------------------------------------

    translation = decode_generated(
        generated,
        id_to_word_fr
    )

    print()

    print("=" * 60)
    print("GENERATED TEXT")
    print("=" * 60)

    print(translation)

    print()

    print("=" * 60)
    print("GREEDY DECODING FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()