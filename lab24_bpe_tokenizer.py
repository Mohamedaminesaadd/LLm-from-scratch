import argparse
import re
from collections import Counter
from pathlib import Path

DATASET_PATH_EN = Path("datasets/sample.txt")
END_TOKEN = "</w>"
MAX_N = 4  # generate patterns of length 1, 2, 3, 4


def read_file(file_path: str) -> str:
    """Read the raw text content of the input file."""
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def tokenize_words(text: str) -> list[str]:
    """Split text into words and append the end-of-word marker to each."""
    words = re.findall(r"\S+", text)
    return [w + END_TOKEN for w in words]


def word_to_symbols(word: str) -> list[str]:
    """
    Turn a word (already ending in </w>) into a list of symbols.
    """
    if word.endswith(END_TOKEN):
        base = word[:-len(END_TOKEN)]
        return list(base) + [END_TOKEN]
    return list(word)


def _ngrams(symbols: list[str], n: int):
    """Yield every consecutive n-length tuple of symbols."""
    for i in range(len(symbols) - n + 1):
        yield tuple(symbols[i:i + n])


def generate_patterns(words: list[str]) -> list[tuple]:
    """
    Generate all unique patterns of length 1..MAX_N.
    """
    seen = set()
    patterns = []

    symbol_sequences = [word_to_symbols(w) for w in words]

    for n in range(1, MAX_N + 1):
        for symbols in symbol_sequences:
            for ng in _ngrams(symbols, n):
                if ng not in seen:
                    seen.add(ng)
                    patterns.append(ng)

    return patterns


def count_patterns(words: list[str], patterns: list[tuple]) -> Counter:
    """
    Count occurrences of each pattern.
    """
    counts = Counter()
    pattern_set = set(patterns)

    symbol_sequences = [word_to_symbols(w) for w in words]

    for n in range(1, MAX_N + 1):
        for symbols in symbol_sequences:
            for ng in _ngrams(symbols, n):
                if ng in pattern_set:
                    counts[ng] += 1

    return counts


def top_patterns(file_path: str, top_n: int = 500):
    """Complete pipeline."""
    text = read_file(file_path)
    words = tokenize_words(text)
    patterns = generate_patterns(words)
    counts = count_patterns(words, patterns)

    return [("".join(pattern), count)
            for pattern, count in counts.most_common(top_n)]


def main():
    parser = argparse.ArgumentParser(
        description="Count top n-gram patterns in a text file."
    )

    parser.add_argument(
        "file_path",
        nargs="?",
        default=str(DATASET_PATH_EN),
        help="Path to the input text file."
    )

    parser.add_argument(
        "--top",
        type=int,
        default=500,
        help="Number of top patterns to display."
    )

    args = parser.parse_args()

    if not Path(args.file_path).exists():
        raise FileNotFoundError(f"File not found: {args.file_path}")

    results = top_patterns(args.file_path, args.top)

    for rank, (pattern, count) in enumerate(results, start=1):
        print(f"{rank:4d}  {pattern:<20} {count}")


if __name__ == "__main__":
    main()