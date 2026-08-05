"""
==========================================================
Lab 24 - Byte Pair Encoding (BPE) Tokenizer
==========================================================

Objective:

1. Load a text corpus.
2. Count word frequencies.
3. Represent words as sequences of characters.
4. Count adjacent token pairs.
5. Find the most frequent pair.
6. Merge the most frequent pair.
7. Repeat the process to learn BPE merge rules.
8. Build a token vocabulary.
9. Tokenize new text using learned merges.
10. Encode tokens into IDs.
11. Decode IDs back into text.

This implementation is intentionally simple and educational.
It implements the core idea behind BPE from scratch.
"""

import re

from collections import Counter
from pathlib import Path


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATASET_PATH = Path(
    "datasets/sample.txt"
)

NUM_MERGES = 100


SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
]


END_OF_WORD = "</w>"


# =================================================
# 1. Load Corpus
# =================================================

def load_corpus(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    return text


# =================================================
# Clean Text
# =================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^\w\s']",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =================================================
# 2. Build Word Frequency
# =================================================

def build_word_frequency(text):

    words = text.split()

    word_frequency = Counter(words)

    return word_frequency


# =================================================
# 3. Initialize Vocabulary
# =================================================

def initialize_vocabulary(
    word_frequency
):

    vocabulary = {}

    for word, frequency in word_frequency.items():

        # Example:
        #
        # low
        #
        # becomes
        #
        # ("l", "o", "w", "</w>")

        symbols = tuple(
            list(word) + [END_OF_WORD]
        )

        vocabulary[symbols] = frequency

    return vocabulary


# =================================================
# 4. Get Pair Frequencies
# =================================================

def get_pair_frequencies(
    vocabulary
):

    pair_frequencies = Counter()

    for symbols, frequency in vocabulary.items():

        # Example:
        #
        # ("l", "o", "w", "</w>")
        #
        # pairs:
        #
        # ("l", "o")
        # ("o", "w")
        # ("w", "</w>")

        for i in range(
            len(symbols) - 1
        ):

            pair = (
                symbols[i],
                symbols[i + 1]
            )

            pair_frequencies[pair] += frequency

    return pair_frequencies


# =================================================
# 5. Find Best Pair
# =================================================

def find_best_pair(
    pair_frequencies
):

    if not pair_frequencies:

        return None

    best_pair = max(
        pair_frequencies,
        key=pair_frequencies.get
    )

    return best_pair


# =================================================
# 6. Merge Pair
# =================================================

def merge_pair(
    pair,
    vocabulary
):

    first, second = pair

    merged_symbol = (
        first + second
    )

    new_vocabulary = {}

    # -------------------------------------------------
    # Process every word representation
    # -------------------------------------------------

    for symbols, frequency in vocabulary.items():

        new_symbols = []

        i = 0

        while i < len(symbols):

            # -----------------------------------------
            # Check whether current + next symbol
            # corresponds to the pair we want to merge
            # -----------------------------------------

            if (
                i < len(symbols) - 1
                and symbols[i] == first
                and symbols[i + 1] == second
            ):

                new_symbols.append(
                    merged_symbol
                )

                # Skip both symbols

                i += 2

            else:

                new_symbols.append(
                    symbols[i]
                )

                i += 1

        new_vocabulary[
            tuple(new_symbols)
        ] = frequency

    return new_vocabulary


# =================================================
# 7. Train BPE
# =================================================

def train_bpe(
    vocabulary,
    num_merges
):

    merges = []

    print()
    print("=" * 60)
    print("BPE TRAINING")
    print("=" * 60)

    for merge_number in range(
        num_merges
    ):

        # -----------------------------------------
        # Count all adjacent pairs
        # -----------------------------------------

        pair_frequencies = (
            get_pair_frequencies(
                vocabulary
            )
        )

        # -----------------------------------------
        # No more pairs
        # -----------------------------------------

        if not pair_frequencies:

            break

        # -----------------------------------------
        # Find most frequent pair
        # -----------------------------------------

        best_pair = find_best_pair(
            pair_frequencies
        )

        frequency = (
            pair_frequencies[
                best_pair
            ]
        )

        # -----------------------------------------
        # Save merge rule
        # -----------------------------------------

        merges.append(
            best_pair
        )

        # -----------------------------------------
        # Apply merge
        # -----------------------------------------

        vocabulary = merge_pair(
            best_pair,
            vocabulary
        )

        # -----------------------------------------
        # Display progress
        # -----------------------------------------

        print(
            f"Merge {merge_number + 1:3d}: "
            f"{best_pair} "
            f"frequency={frequency}"
        )

    return (
        vocabulary,
        merges
    )


# =================================================
# 8. Build Token Vocabulary
# =================================================

def build_token_vocabulary(
    vocabulary
):

    tokens = set()

    # -------------------------------------------------
    # Collect every learned token
    # -------------------------------------------------

    for symbols in vocabulary:

        for symbol in symbols:

            tokens.add(symbol)

    # -------------------------------------------------
    # Special tokens first
    # -------------------------------------------------

    token_to_id = {}

    id_to_token = {}

    current_id = 0

    for token in SPECIAL_TOKENS:

        token_to_id[token] = current_id

        id_to_token[current_id] = token

        current_id += 1

    # -------------------------------------------------
    # Add BPE tokens
    # -------------------------------------------------

    for token in sorted(tokens):

        if token not in token_to_id:

            token_to_id[token] = current_id

            id_to_token[current_id] = token

            current_id += 1

    return (
        token_to_id,
        id_to_token
    )


# =================================================
# Apply Learned BPE Merges to One Word
# =================================================

def apply_bpe_to_word(
    word,
    merges
):

    # -------------------------------------------------
    # Start from characters
    # -------------------------------------------------

    symbols = (
        list(word)
        + [END_OF_WORD]
    )

    # -------------------------------------------------
    # Apply merges in learned order
    # -------------------------------------------------

    for first, second in merges:

        new_symbols = []

        i = 0

        while i < len(symbols):

            if (
                i < len(symbols) - 1
                and symbols[i] == first
                and symbols[i + 1] == second
            ):

                new_symbols.append(
                    first + second
                )

                i += 2

            else:

                new_symbols.append(
                    symbols[i]
                )

                i += 1

        symbols = new_symbols

    return symbols


# =================================================
# 9. Tokenize
# =================================================

def tokenize(
    text,
    merges
):

    text = clean_text(text)

    words = text.split()

    tokens = []

    for word in words:

        word_tokens = (
            apply_bpe_to_word(
                word,
                merges
            )
        )

        tokens.extend(
            word_tokens
        )

    return tokens


# =================================================
# 10. Encode
# =================================================

def encode(
    text,
    merges,
    token_to_id,
    add_special_tokens=True
):

    tokens = tokenize(
        text,
        merges
    )

    unk_id = token_to_id[
        "<UNK>"
    ]

    token_ids = []

    # -------------------------------------------------
    # BOS
    # -------------------------------------------------

    if add_special_tokens:

        token_ids.append(
            token_to_id["<BOS>"]
        )

    # -------------------------------------------------
    # BPE tokens
    # -------------------------------------------------

    for token in tokens:

        token_id = token_to_id.get(
            token,
            unk_id
        )

        token_ids.append(
            token_id
        )

    # -------------------------------------------------
    # EOS
    # -------------------------------------------------

    if add_special_tokens:

        token_ids.append(
            token_to_id["<EOS>"]
        )

    return token_ids


# =================================================
# 11. Decode
# =================================================

def decode(
    token_ids,
    id_to_token
):

    tokens = []

    # -------------------------------------------------
    # IDs -> tokens
    # -------------------------------------------------

    for token_id in token_ids:

        token = id_to_token.get(
            token_id,
            "<UNK>"
        )

        # Ignore special tokens

        if token in {
            "<PAD>",
            "<BOS>",
            "<EOS>"
        }:

            continue

        tokens.append(token)

    # -------------------------------------------------
    # Join BPE tokens
    # -------------------------------------------------

    text = "".join(tokens)

    # -------------------------------------------------
    # </w> represents a word boundary
    # -------------------------------------------------

    text = text.replace(
        END_OF_WORD,
        " "
    )

    return text.strip()


# =================================================
# Display Vocabulary Example
# =================================================

def display_vocabulary(
    vocabulary,
    limit=20
):

    print()
    print("=" * 60)
    print("BPE VOCABULARY EXAMPLES")
    print("=" * 60)

    for index, (
        symbols,
        frequency
    ) in enumerate(
        vocabulary.items()
    ):

        print(
            f"{symbols} "
            f"frequency={frequency}"
        )

        if index + 1 >= limit:

            break


# =================================================
# Display Merge Rules
# =================================================

def display_merges(
    merges,
    limit=30
):

    print()
    print("=" * 60)
    print("LEARNED MERGE RULES")
    print("=" * 60)

    for index, pair in enumerate(
        merges[:limit]
    ):

        print(
            f"{index + 1:3d}: "
            f"{pair[0]} + {pair[1]} "
            f"-> {pair[0] + pair[1]}"
        )


# =================================================
# 12. Main
# =================================================

def main():

    print("=" * 60)
    print("LAB 24 - BPE TOKENIZER")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Load Corpus
    # -------------------------------------------------

    text = load_corpus(
        DATASET_PATH
    )

    text = clean_text(
        text
    )

    print()
    print("Corpus characters:")
    print(len(text))

    # -------------------------------------------------
    # 2. Word Frequencies
    # -------------------------------------------------

    word_frequency = (
        build_word_frequency(
            text
        )
    )

    print()
    print("Unique words:")
    print(len(word_frequency))

    print()
    print("Most common words:")

    for word, frequency in (
        word_frequency.most_common(10)
    ):

        print(
            f"{word:20} "
            f"{frequency}"
        )

    # -------------------------------------------------
    # 3. Initialize Vocabulary
    # -------------------------------------------------

    vocabulary = (
        initialize_vocabulary(
            word_frequency
        )
    )

    print()
    print("=" * 60)
    print("INITIAL CHARACTER VOCABULARY")
    print("=" * 60)

    for index, (
        symbols,
        frequency
    ) in enumerate(
        vocabulary.items()
    ):

        print(
            symbols,
            "frequency=",
            frequency
        )

        if index >= 9:
            break

    # -------------------------------------------------
    # 4 -> 7. Train BPE
    # -------------------------------------------------

    (
        trained_vocabulary,
        merges

    ) = train_bpe(
        vocabulary,
        NUM_MERGES
    )

    # -------------------------------------------------
    # Display learned merges
    # -------------------------------------------------

    display_merges(
        merges
    )

    # -------------------------------------------------
    # Display vocabulary
    # -------------------------------------------------

    display_vocabulary(
        trained_vocabulary
    )

    # -------------------------------------------------
    # 8. Build Token Vocabulary
    # -------------------------------------------------

    (
        token_to_id,
        id_to_token

    ) = build_token_vocabulary(
        trained_vocabulary
    )

    print()
    print("=" * 60)
    print("TOKEN VOCABULARY")
    print("=" * 60)

    print(
        "Vocabulary size:",
        len(token_to_id)
    )

    print()

    print("Special tokens:")

    for token in SPECIAL_TOKENS:

        print(
            f"{token:8} -> "
            f"{token_to_id[token]}"
        )

    # -------------------------------------------------
    # Test Tokenizer
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("TOKENIZER TEST")
    print("=" * 60)

    sentence = input(
        "\nEnter a sentence: "
    )

    # -------------------------------------------------
    # 9. Tokenize
    # -------------------------------------------------

    tokens = tokenize(
        sentence,
        merges
    )

    print()
    print("BPE Tokens:")

    print(tokens)

    # -------------------------------------------------
    # 10. Encode
    # -------------------------------------------------

    token_ids = encode(
        sentence,
        merges,
        token_to_id
    )

    print()
    print("Token IDs:")

    print(token_ids)

    # -------------------------------------------------
    # Display Token -> ID
    # -------------------------------------------------

    print()
    print("Token / ID:")

    for token_id in token_ids:

        print(
            f"{token_id:4d} "
            f"-> "
            f"{id_to_token[token_id]}"
        )

    # -------------------------------------------------
    # 11. Decode
    # -------------------------------------------------

    decoded_text = decode(
        token_ids,
        id_to_token
    )

    print()
    print("Decoded:")

    print(decoded_text)

    print()
    print("=" * 60)
    print("LAB 24 FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()