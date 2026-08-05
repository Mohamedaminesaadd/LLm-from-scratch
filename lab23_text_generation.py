"""
==========================================================
Lab 23 - Text Generation
==========================================================

Objective:

1. Load the trained Transformer.
2. Accept raw English text from the user.
3. Clean and encode the source text.
4. Generate target tokens using:
      - Greedy Decoding
      - Beam Search
5. Convert generated token IDs back to text.
6. Create a clean text-generation interface.

Pipeline:

Raw Text
   ↓
Clean Text
   ↓
Token IDs
   ↓
Transformer
   ↓
Greedy / Beam Search
   ↓
Generated Token IDs
   ↓
Generated Text
"""

import torch

# -------------------------------------------------
# Reuse Lab 21
# -------------------------------------------------

from lab21_greedy_decoding import (
    load_model,
    encode_source,
    greedy_decode,
    decode_generated,
)

# -------------------------------------------------
# Reuse Lab 22
# -------------------------------------------------

from lab22_beam_search import (
    beam_search_decode,
    decode_sequence,
)


# -------------------------------------------------
# Configuration
# -------------------------------------------------

MAX_GENERATION_LENGTH = 20
BEAM_SIZE = 3


# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------------------------------
# Text Generation Function
# -------------------------------------------------

def generate_text(
    model,
    text,
    word_to_id_en,
    word_to_id_fr,
    id_to_word_fr,
    strategy="greedy",
    max_length=20,
    beam_size=3
):
    """
    Generate French text from an English sentence.

    Parameters
    ----------
    model:
        Trained Transformer.

    text:
        Raw English sentence.

    word_to_id_en:
        English vocabulary.

    word_to_id_fr:
        French vocabulary.

    id_to_word_fr:
        Reverse French vocabulary.

    strategy:
        "greedy" or "beam".

    max_length:
        Maximum number of generated tokens.

    beam_size:
        Number of beams used by Beam Search.

    Returns
    -------
    generated_text:
        Generated French sentence.
    """

    # -------------------------------------------------
    # Encode Source Text
    # -------------------------------------------------

    source = encode_source(
        text,
        word_to_id_en
    )

    # -------------------------------------------------
    # Greedy Decoding
    # -------------------------------------------------

    if strategy == "greedy":

        generated = greedy_decode(
            model=model,
            source=source,
            word_to_id_fr=word_to_id_fr,
            max_length=max_length
        )

        # Convert tensor tokens -> text

        generated_text = decode_generated(
            generated,
            id_to_word_fr
        )

        return generated_text

    # -------------------------------------------------
    # Beam Search
    # -------------------------------------------------

    elif strategy == "beam":

        (
            best_sequence,
            best_score,
            beams

        ) = beam_search_decode(
            model=model,
            source=source,
            word_to_id_fr=word_to_id_fr,
            beam_size=beam_size,
            max_length=max_length
        )

        # Convert token IDs -> text

        generated_text = decode_sequence(
            best_sequence,
            id_to_word_fr
        )

        return generated_text

    # -------------------------------------------------
    # Invalid Strategy
    # -------------------------------------------------

    else:

        raise ValueError(
            "Unknown generation strategy: "
            f"{strategy}\n"
            "Available strategies: "
            "'greedy', 'beam'"
        )


# -------------------------------------------------
# Compare Generation Strategies
# -------------------------------------------------

def compare_strategies(
    model,
    text,
    word_to_id_en,
    word_to_id_fr,
    id_to_word_fr
):

    print()
    print("=" * 60)
    print("GENERATION COMPARISON")
    print("=" * 60)

    # -------------------------------------------------
    # Greedy
    # -------------------------------------------------

    print()
    print("Running Greedy Decoding...")
    print()

    greedy_text = generate_text(
        model=model,
        text=text,
        word_to_id_en=word_to_id_en,
        word_to_id_fr=word_to_id_fr,
        id_to_word_fr=id_to_word_fr,
        strategy="greedy",
        max_length=MAX_GENERATION_LENGTH
    )

    # -------------------------------------------------
    # Beam
    # -------------------------------------------------

    print()
    print("Running Beam Search...")
    print()

    beam_text = generate_text(
        model=model,
        text=text,
        word_to_id_en=word_to_id_en,
        word_to_id_fr=word_to_id_fr,
        id_to_word_fr=id_to_word_fr,
        strategy="beam",
        max_length=MAX_GENERATION_LENGTH,
        beam_size=BEAM_SIZE
    )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print()
    print("SOURCE:")
    print(text)

    print()
    print("GREEDY:")
    print(greedy_text)

    print()
    print("BEAM SEARCH:")
    print(beam_text)


# -------------------------------------------------
# Interactive Generation
# -------------------------------------------------

def interactive_generation(
    model,
    word_to_id_en,
    word_to_id_fr,
    id_to_word_fr
):

    print()
    print("=" * 60)
    print("INTERACTIVE TEXT GENERATION")
    print("=" * 60)

    print()
    print("Available strategies:")
    print("1 -> Greedy Decoding")
    print("2 -> Beam Search")
    print("3 -> Compare both")
    print("q -> Quit")

    # -------------------------------------------------
    # Interactive Loop
    # -------------------------------------------------

    while True:

        print()
        print("-" * 60)

        sentence = input(
            "Enter an English sentence (q to quit): "
        ).strip()

        # -------------------------------------------------
        # Quit
        # -------------------------------------------------

        if sentence.lower() == "q":

            print()
            print("Stopping generation.")

            break

        # -------------------------------------------------
        # Empty Input
        # -------------------------------------------------

        if not sentence:

            print("Please enter a sentence.")

            continue

        # -------------------------------------------------
        # Strategy
        # -------------------------------------------------

        print()

        strategy_choice = input(
            "Strategy [1=greedy, 2=beam, 3=compare]: "
        ).strip()

        # -------------------------------------------------
        # Greedy
        # -------------------------------------------------

        if strategy_choice == "1":

            generated_text = generate_text(
                model=model,
                text=sentence,
                word_to_id_en=word_to_id_en,
                word_to_id_fr=word_to_id_fr,
                id_to_word_fr=id_to_word_fr,
                strategy="greedy",
                max_length=MAX_GENERATION_LENGTH
            )

            print()
            print("=" * 60)
            print("GREEDY RESULT")
            print("=" * 60)

            print()
            print("Source:")
            print(sentence)

            print()
            print("Generated:")
            print(generated_text)

        # -------------------------------------------------
        # Beam Search
        # -------------------------------------------------

        elif strategy_choice == "2":

            generated_text = generate_text(
                model=model,
                text=sentence,
                word_to_id_en=word_to_id_en,
                word_to_id_fr=word_to_id_fr,
                id_to_word_fr=id_to_word_fr,
                strategy="beam",
                max_length=MAX_GENERATION_LENGTH,
                beam_size=BEAM_SIZE
            )

            print()
            print("=" * 60)
            print("BEAM SEARCH RESULT")
            print("=" * 60)

            print()
            print("Source:")
            print(sentence)

            print()
            print("Generated:")
            print(generated_text)

        # -------------------------------------------------
        # Compare
        # -------------------------------------------------

        elif strategy_choice == "3":

            compare_strategies(
                model=model,
                text=sentence,
                word_to_id_en=word_to_id_en,
                word_to_id_fr=word_to_id_fr,
                id_to_word_fr=id_to_word_fr
            )

        # -------------------------------------------------
        # Invalid Choice
        # -------------------------------------------------

        else:

            print(
                "Invalid strategy. "
                "Choose 1, 2, or 3."
            )


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("LAB 23 - TEXT GENERATION")
    print("=" * 60)

    print()
    print("Device:", device)
    print()

    # -------------------------------------------------
    # Load Model
    # -------------------------------------------------

    (
        model,
        word_to_id_en,
        id_to_word_en,
        word_to_id_fr,
        id_to_word_fr

    ) = load_model()

    # -------------------------------------------------
    # Check Special Tokens
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

    # -------------------------------------------------
    # Verify BOS / EOS
    # -------------------------------------------------

    if "<BOS>" not in word_to_id_fr:

        print()
        print(
            "ERROR: <BOS> is missing "
            "from the target vocabulary."
        )

        return

    if "<EOS>" not in word_to_id_fr:

        print()
        print(
            "ERROR: <EOS> is missing "
            "from the target vocabulary."
        )

        return

    # -------------------------------------------------
    # Start Interactive Generation
    # -------------------------------------------------

    interactive_generation(
        model=model,
        word_to_id_en=word_to_id_en,
        word_to_id_fr=word_to_id_fr,
        id_to_word_fr=id_to_word_fr
    )

    print()
    print("=" * 60)
    print("LAB 23 FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()