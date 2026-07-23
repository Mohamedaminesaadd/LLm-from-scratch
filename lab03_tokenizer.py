"""
=================================================
Lab 03 - Simple Tokenizer
=================================================

Objective:
1. Build a vocabulary.
2. Encode text into token IDs.
3. Decode token IDs back into text.
4. Handle unknown words.
5. Add special tokens (<BOS>, <EOS>).
"""


from pathlib import Path
from collections import Counter
import re


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

def load_text(text_path:Path):
    """ fonction for loading text from file"""
    with open(text_path,"r",encoding="utf-8") as f :
        return f.read()



#---------------------------------
# clean text
#----------------------------------------

def clean_text(text:str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]","",text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



# -------------------------------------------------
# Build Vocabulary
# -------------------------------------------------


def build_vocabulary(text:str):
    tokens = text.split()
    counter = Counter(tokens) #dict { mot:count } in the list 

    word_to_id = {}
    id_to_word = {}
    idx =0

    for token in SPECIAL_TOKENS :
        word_to_id[token] = SPECIAL_TOKENS[idx]
        id_to_word[idx] = token
        idx+=1

    for word in sorted(counter.keys()): #  list keys with alphabitique order 
        word_to_id[word] = idx
        id_to_word[idx] = word
        idx+=1
    
    return word_to_id, id_to_word


# -------------------------------------------------
# Tokenizer
# -------------------------------------------------

class SimpleTokenizer:

    def __init__(self, word_to_id, id_to_word):

        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

    def encode(self, sentence: str):

        sentence = clean_text(sentence)

        words = sentence.split()

        ids = [self.word_to_id["<BOS>"]]

        for word in words:
            ids.append(
                self.word_to_id.get(
                    word,
                    self.word_to_id["<UNK>"]
                )
            )

        ids.append(self.word_to_id["<EOS>"])

        return ids

    def decode(self, ids):

        words = []

        for token_id in ids:

            word = self.id_to_word.get(
                token_id,
                "<UNK>"
            )

            if word in SPECIAL_TOKENS:
                continue

            words.append(word)

        return " ".join(words)



# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    text = load_text(DATASET_PATH)

    text = clean_text(text)

    word_to_id, id_to_word = build_vocabulary(text)

    tokenizer = SimpleTokenizer(
        word_to_id,
        id_to_word
    )

    sentence = " Adding training data decreases the variance"

    ids = tokenizer.encode(sentence)

    reconstructed = tokenizer.decode(ids)

    print("=" * 50)
    print("Original Sentence")
    print("=" * 50)
    print(sentence)

    print()

    print("=" * 50)
    print("Encoded")
    print("=" * 50)
    print(ids)

    print()

    print("=" * 50)
    print("Decoded")
    print("=" * 50)
    print(reconstructed)

    print()

    unknown = " Adding training data --    Ecreases the variance"

    print("=" * 50)
    print("Unknown Word Example")
    print("=" * 50)
    print(unknown)
    print(tokenizer.encode(unknown))


if __name__ == "__main__":
    main()     