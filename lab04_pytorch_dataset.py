"""
=================================================
Lab 04 - PyTorch Dataset
=================================================

Objective:
1. Load a text dataset.
2. Build a vocabulary.
3. Encode the text into token IDs.
4. Create input-target training pairs.
5. Build a custom PyTorch Dataset.
6. Use a DataLoader to create mini-batches.
"""


from pathlib import Path
from collections import Counter
import re

import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATASET_PATH = Path("datasets/sample.txt")

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
]

SEQUENCE_LENGTH = 5
BATCH_SIZE = 2


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

def load_text(path: Path) -> str:

    with open(path, "r", encoding="utf-8") as file:
        return file.read()



def clean_text(text: str) -> str:

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()



# -------------------------------------------------
# Build Vocabulary
# -------------------------------------------------

def build_vocabulary(text: str):

    counter = Counter(text.split())

    word_to_id = {}
    id_to_word = {}

    idx = 0

    for token in SPECIAL_TOKENS:
        word_to_id[token] = idx
        id_to_word[idx] = token
        idx += 1

    for word in sorted(counter.keys()):
        word_to_id[word] = idx
        id_to_word[idx] = word
        idx += 1

    return word_to_id, id_to_word


# -------------------------------------------------
# Encode Text
# -------------------------------------------------

def encode(text, word_to_id):

    tokens = text.split()

    ids = []

    for token in tokens:
        ids.append(
            word_to_id.get(
                token,
                word_to_id["<UNK>"]
            )
        )

    return ids



# -------------------------------------------------
# Custom Dataset
# -------------------------------------------------

class TextDataset(Dataset):

    def __init__(self, token_ids, sequence_length):

        self.inputs = []
        self.targets = []

        for i in range(len(token_ids) - sequence_length):

            x = token_ids[i:i + sequence_length]

            y = token_ids[i + 1:i + sequence_length + 1]

            self.inputs.append(torch.tensor(x))
            self.targets.append(torch.tensor(y))

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        return self.inputs[index], self.targets[index]

# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    text = load_text(DATASET_PATH)

    text = clean_text(text)

    word_to_id, id_to_word = build_vocabulary(text)

    token_ids = encode(text, word_to_id)

    dataset = TextDataset(
        token_ids,
        SEQUENCE_LENGTH
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("=" * 50)
    print("Dataset Information")
    print("=" * 50)

    print("Vocabulary Size :", len(word_to_id))
    print("Total Tokens    :", len(token_ids))
    print("Training Samples:", len(dataset))

    print()

    print("=" * 50)
    print("One Batch")
    print("=" * 50)

    for inputs, targets in dataloader:

        print("Input Shape :", inputs.shape)
        print("Target Shape:", targets.shape)

        print()

        print("Inputs")
        print(inputs)

        print()

        print("Targets")
        print(targets)

        break


if __name__ == "__main__":
    main()