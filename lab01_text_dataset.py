"""
=================================================
Lab 01 - Load and Explore a Text Dataset
=================================================

Objective:
1. Load a text file.
2. Clean the text.
3. Split it into sentences.
4. Display basic statistics.

Dataset:
datasets/sample.txt
"""

from pathlib import Path
import re


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATASET_PATH = Path("datasets/sample.txt")


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

def load_text(file_path: Path) -> str:
    """Load a text file."""

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


# -------------------------------------------------
# Clean Text
# -------------------------------------------------

def clean_text(text: str) -> str:
    """
    Basic cleaning.

    - lowercase
    - remove multiple spaces
    - remove extra newlines
    """

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -------------------------------------------------
# Split into Sentences
# -------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    """

    sentences = re.split(r"[.!?]+", text)

    return [s.strip() for s in sentences if s.strip()]


# -------------------------------------------------
# Dataset Statistics
# -------------------------------------------------

def dataset_statistics(text: str, sentences: list[str]):

    words = text.split()

    print("=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    print(f"Characters : {len(text)}")
    print(f"Words      : {len(words)}")
    print(f"Sentences  : {len(sentences)}")

    print()


# -------------------------------------------------
# Preview
# -------------------------------------------------

def preview(sentences: list[str], n: int = 5):

    print("=" * 50)
    print("FIRST SENTENCES")
    print("=" * 50)

    for i, sentence in enumerate(sentences[:n], start=1):
        print(f"{i}. {sentence}")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("Loading dataset...\n")

    text = load_text(DATASET_PATH)

    text = clean_text(text)

    sentences = split_sentences(text)

    dataset_statistics(text, sentences)

    preview(sentences)


if __name__ == "__main__":
    main()