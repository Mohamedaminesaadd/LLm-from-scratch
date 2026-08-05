"""
==========================================================
Lab 22 - Beam Search Decoding
==========================================================

Objective:

1. Load the trained Transformer checkpoint.
2. Encode an English source sentence.
3. Start generation with <BOS>.
4. Maintain multiple candidate sequences.
5. Expand each candidate using the best next tokens.
6. Score candidates using log probabilities.
7. Keep only the best K candidates.
8. Stop completed sequences when <EOS> is generated.
9. Return the best generated translation.

Lab 21:
    Greedy Search
    -> keep 1 candidate

Lab 22:
    Beam Search
    -> keep K candidates
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from lab06_positional_encoding import (
    clean_text,
    encode,
)

from lab18_transformer_model import TransformerModel


# -------------------------------------------------
# Configuration
# -------------------------------------------------

MODEL_PATH = Path("models/transformer.pth")

BEAM_SIZE = 3
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
            "Run Lab 19 first."
        )

    # -------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    # -------------------------------------------------
    # Recover architecture configuration
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
    # Load trained weights
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
# Encode Source
# -------------------------------------------------

def encode_source(
    sentence,
    word_to_id_en
):

    # -------------------------------------------------
    # Clean sentence
    # -------------------------------------------------

    sentence = clean_text(
        sentence
    )

    # -------------------------------------------------
    # Encode sentence
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
    # Convert to Tensor
    # -------------------------------------------------

    source = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device
    )

    # Add batch dimension
    #
    # [seq]
    #
    # becomes
    #
    # [1, seq]

    source = source.unsqueeze(0)

    return source


# -------------------------------------------------
# Beam Search
# -------------------------------------------------

def beam_search_decode(
    model,
    source,
    word_to_id_fr,
    beam_size=3,
    max_length=20
):

    # -------------------------------------------------
    # Check special tokens
    # -------------------------------------------------

    if "<BOS>" not in word_to_id_fr:

        raise ValueError(
            "<BOS> token does not exist "
            "in the target vocabulary."
        )

    if "<EOS>" not in word_to_id_fr:

        raise ValueError(
            "<EOS> token does not exist "
            "in the target vocabulary."
        )

    bos_id = word_to_id_fr[
        "<BOS>"
    ]

    eos_id = word_to_id_fr[
        "<EOS>"
    ]

    # -------------------------------------------------
    # Initial Beam
    # -------------------------------------------------
    #
    # Each beam contains:
    #
    # (
    #     sequence,
    #     score,
    #     finished
    # )
    #
    # Initially:
    #
    # sequence = [BOS]
    # score = 0
    # finished = False
    # -------------------------------------------------

    beams = [
        (
            [bos_id],
            0.0,
            False
        )
    ]

    print("=" * 60)
    print("BEAM SEARCH")
    print("=" * 60)

    print("Beam size :", beam_size)
    print("Max length:", max_length)
    print()

    # -------------------------------------------------
    # Disable gradients
    # -------------------------------------------------

    with torch.no_grad():

        # -------------------------------------------------
        # Generation Loop
        # -------------------------------------------------

        for step in range(max_length):

            print()
            print("-" * 60)
            print(f"STEP {step + 1}")
            print("-" * 60)

            candidates = []

            # -------------------------------------------------
            # Expand every current beam
            # -------------------------------------------------

            for (
                sequence,
                score,
                finished
            ) in beams:

                # -----------------------------------------
                # If EOS was already generated,
                # don't expand this sequence.
                # -----------------------------------------

                if finished:

                    candidates.append(
                        (
                            sequence,
                            score,
                            True
                        )
                    )

                    continue

                # -----------------------------------------
                # Convert sequence to tensor
                # -----------------------------------------

                decoder_input = torch.tensor(
                    sequence,
                    dtype=torch.long,
                    device=device
                )

                # [seq]
                #
                # ->
                #
                # [1, seq]

                decoder_input = (
                    decoder_input.unsqueeze(0)
                )

                # -----------------------------------------
                # Transformer Forward Pass
                # -----------------------------------------

                logits, _, _, _ = model(
                    source,
                    decoder_input
                )

                # logits:
                #
                # [1,
                #  decoder_length,
                #  target_vocab_size]

                # -----------------------------------------
                # Get last position
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
                # Convert logits -> log probabilities
                # -----------------------------------------

                log_probs = F.log_softmax(
                    last_token_logits,
                    dim=-1
                )

                # -----------------------------------------
                # Get Top-K next tokens
                # -----------------------------------------

                top_scores, top_tokens = torch.topk(
                    log_probs,
                    k=beam_size,
                    dim=-1
                )

                # Remove batch dimension

                top_scores = top_scores[0]

                top_tokens = top_tokens[0]

                # -----------------------------------------
                # Create new candidate sequences
                # -----------------------------------------

                for i in range(beam_size):

                    token_id = (
                        top_tokens[i].item()
                    )

                    token_score = (
                        top_scores[i].item()
                    )

                    # New sequence

                    new_sequence = (
                        sequence + [token_id]
                    )

                    # Accumulate log probability

                    new_score = (
                        score + token_score
                    )

                    # Check EOS

                    new_finished = (
                        token_id == eos_id
                    )

                    # Add candidate

                    candidates.append(
                        (
                            new_sequence,
                            new_score,
                            new_finished
                        )
                    )

            # -------------------------------------------------
            # Sort candidates
            # -------------------------------------------------
            #
            # Highest score first.
            #
            # Remember:
            #
            # log probabilities are normally negative.
            #
            # -1.2 > -5.7
            #
            # therefore -1.2 is better.
            # -------------------------------------------------

            candidates.sort(
                key=lambda x: x[1],
                reverse=True
            )

            # -------------------------------------------------
            # Keep only best K
            # -------------------------------------------------

            beams = candidates[
                :beam_size
            ]

            # -------------------------------------------------
            # Display beams
            # -------------------------------------------------

            for beam_index, (
                sequence,
                score,
                finished
            ) in enumerate(beams):

                print(
                    f"Beam {beam_index + 1}: "
                    f"{sequence} "
                    f"score={score:.4f} "
                    f"finished={finished}"
                )

            # -------------------------------------------------
            # Stop if ALL beams finished
            # -------------------------------------------------

            all_finished = all(
                finished
                for _, _, finished in beams
            )

            if all_finished:

                print()
                print(
                    "All beams generated <EOS>."
                )

                print(
                    "Stopping beam search."
                )

                break

    # -------------------------------------------------
    # Final sorting
    # -------------------------------------------------

    beams.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # -------------------------------------------------
    # Best Beam
    # -------------------------------------------------

    best_sequence = beams[0][0]

    best_score = beams[0][1]

    return (
        best_sequence,
        best_score,
        beams
    )


# -------------------------------------------------
# Decode Sequence
# -------------------------------------------------

def decode_sequence(
    sequence,
    id_to_word_fr
):

    words = []

    for token_id in sequence:

        word = id_to_word_fr.get(
            token_id,
            "<UNK>"
        )

        # Ignore BOS

        if word == "<BOS>":
            continue

        # Stop at EOS

        if word == "<EOS>":
            break

        # Ignore padding

        if word == "<PAD>":
            continue

        words.append(word)

    return " ".join(words)


# -------------------------------------------------
# Display Source Tokens
# -------------------------------------------------

def display_source(
    source,
    id_to_word_en
):

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
# Display Final Beams
# -------------------------------------------------

def display_final_beams(
    beams,
    id_to_word_fr
):

    print()
    print("=" * 60)
    print("FINAL BEAMS")
    print("=" * 60)

    for index, (
        sequence,
        score,
        finished
    ) in enumerate(beams):

        text = decode_sequence(
            sequence,
            id_to_word_fr
        )

        print()

        print(
            f"Beam {index + 1}"
        )

        print(
            "Token IDs:",
            sequence
        )

        print(
            f"Score: {score:.4f}"
        )

        print(
            "Finished:",
            finished
        )

        print(
            "Text:",
            text
        )


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("LAB 22 - BEAM SEARCH")
    print("=" * 60)

    print()

    # -------------------------------------------------
    # Load Transformer
    # -------------------------------------------------

    (
        model,
        word_to_id_en,
        id_to_word_en,
        word_to_id_fr,
        id_to_word_fr

    ) = load_model()

    # -------------------------------------------------
    # Display special tokens
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

        token_id = word_to_id_fr.get(
            token,
            "NOT FOUND"
        )

        print(
            f"{token:8} -> {token_id}"
        )

    print()

    # -------------------------------------------------
    # User Input
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

    print(
        "Source shape:",
        source.shape
    )

    print(
        "Source IDs:",
        source
    )

    # -------------------------------------------------
    # Display Source
    # -------------------------------------------------

    display_source(
        source,
        id_to_word_en
    )

    # -------------------------------------------------
    # Beam Search
    # -------------------------------------------------

    (
        best_sequence,
        best_score,
        beams

    ) = beam_search_decode(
        model=model,
        source=source,
        word_to_id_fr=word_to_id_fr,
        beam_size=BEAM_SIZE,
        max_length=MAX_GENERATION_LENGTH
    )

    # -------------------------------------------------
    # Display Final Beams
    # -------------------------------------------------

    display_final_beams(
        beams,
        id_to_word_fr
    )

    # -------------------------------------------------
    # Decode Best Sequence
    # -------------------------------------------------

    translation = decode_sequence(
        best_sequence,
        id_to_word_fr
    )

    # -------------------------------------------------
    # Best Result
    # -------------------------------------------------

    print()

    print("=" * 60)
    print("BEST BEAM")
    print("=" * 60)

    print(
        "Token IDs:",
        best_sequence
    )

    print(
        f"Score: {best_score:.4f}"
    )

    print(
        "Generated text:",
        translation
    )

    print()

    print("=" * 60)
    print("BEAM SEARCH FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()