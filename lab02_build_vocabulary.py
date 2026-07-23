"""
====================================================
lab -02 build vocabulary
====================================================
objective :
    1-load a text dataset.
    2-clean the text
    3-tokenize into the words 
    4-count word frequencies
    5-build a vocabulary (worf -> id)
    6-build the reverce vocabulary(id -> word)
"""

from pathlib import Path
from collections import Counter
import re


#------------------------------------------------
#configuration
#------------------------------------------------

DATASET_PATH = Path("datasets/sample.txt")

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
]


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

def load_dataset(data_path:Path):
    """ load dataset samples.text"""
    with open(data_path,"r",encoding="utf-8") as file:
        return file.read()


#---------------------------------------------------
#clean text 
#---------------------------------------------------

def clean_text(text:str)-> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



# -------------------------------------------------
# Tokenize
# -------------------------------------------------

def tokenize(text:str) -> list[str]:
    return text.split()

#----------------------------------------------------
#count the word 
#--------------------------------------------

def count_word(text:str) -> Counter:
    return Counter(text)


#---------------------------------------------------
#bluid vocabulary 
#---------------------------------------------------
def build_vocabulary(counter: Counter):

    word_to_id = {}

    id_to_word = {}

    current_id = 0

    # Add special tokens first
    for token in SPECIAL_TOKENS:

        word_to_id[token] = current_id
        id_to_word[current_id] = token

        current_id += 1

    # Add dataset words
    for word in sorted(counter.keys()):

        word_to_id[word] = current_id
        id_to_word[current_id] = word

        current_id += 1

    return word_to_id, id_to_word

# -------------------------------------------------
# Display Vocabulary
# -------------------------------------------------


def show_vocabulary(word_to_id, limit=20):

    print("=" * 50)
    print("VOCABULARY")
    print("=" * 50)

    for i, (word, idx) in enumerate(word_to_id.items()):

        print(f"{idx:3} -> {word}")

        if i + 1 >= limit:
            break


def main():
    print("loading dataset ...\n")
    text = load_dataset(DATASET_PATH)
    text = clean_text(text)
    tokens = tokenize(text)
    counter = count_word(tokens)
    word_to_id, id_to_word = build_vocabulary(counter)
    print(f"Total words      : {len(tokens)}")
    print(f"Unique words     : {len(counter)}")
    print(f"Vocabulary size  : {len(word_to_id)}\n")

    show_vocabulary(word_to_id)

    print("\nExample Lookup")
    print("-" * 30)

    print("Word 'transformer' ->", word_to_id.get("transformer", word_to_id["<UNK>"]))

    print("ID 0 ->", id_to_word[0])


if __name__ == "__main__":
    main()