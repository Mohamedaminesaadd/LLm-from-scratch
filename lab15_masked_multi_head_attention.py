import math
import re
from collections import Counter
from pathlib import Path

from lab06_positional_encoding import PositionalEncoding, TextDataset, build_vocabulary, clean_text, encode, load_text
from lab13_encoder_block import AddNorm, FeedForward, MultiHeadSelfAttention
import torch
import torch.nn as nn
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
EMBEDDING_DIM = 100
HEAD_NUMBER = 4
FF_HIDDEN_DIM = 4 * EMBEDDING_DIM
DROPOUT = 0.1



# -------------------------------------------------
# Transformer Encoder Block
# -------------------------------------------------

class TransformerBlock(nn.Module):
    """
    x -> MultiHeadSelfAttention -> AddNorm -> FeedForward -> AddNorm
    """

    def __init__(self, embedding_dim, head_number, hidden_dim, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(embedding_dim, head_number)
        self.add_norm_1 = AddNorm(embedding_dim, dropout)

        self.feed_forward = FeedForward(embedding_dim, hidden_dim, dropout)
        self.add_norm_2 = AddNorm(embedding_dim, dropout)

    def forward(self, x):

        # Sub-layer 1 : self-attention + residual + norm
        attention_output, attention_weights = self.attention(x)
        x = self.add_norm_1(x, attention_output)

        # Sub-layer 2 : feed forward + residual + norm
        feed_forward_output = self.feed_forward(x)
        x = self.add_norm_2(x, feed_forward_output)

        return x, attention_weights




class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, embedding_dim)
        """

        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.embedding_dim)

        # Causal mask
        seq_len = x.size(1)

        mask = torch.triu(
            torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=x.device
            ),
            diagonal=1
        )

        # Apply mask
        scores = scores + mask

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # Output
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


#-----------------------------------------------
# masked multi head self attention 
#------------------------------------------------
 
class MaskedMultiHeadSelfAttention(nn.Module):
    """
    Identique à MultiHeadSelfAttention, mais chaque position ne peut
    regarder que les positions <= à elle (masque causal triangulaire).
    """
 
    def __init__(self, embedding_dim, head_number):
        super().__init__()
 
        assert embedding_dim % head_number == 0, \
            "embedding_dim must be divisible by head_number"
 
        self.embedding_dim = embedding_dim
        self.head_number = head_number
        self.head_dim = embedding_dim // head_number
 
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
 
        self.out = nn.Linear(embedding_dim, embedding_dim)
 
    @staticmethod
    def build_causal_mask(seq_len, device):
        # True = position interdite (futur)
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
 
    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)
 
        batch_size, seq_len, _ = x.size()
 
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
 
        # Split into heads -> (batch, head, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.head_number, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.head_number, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.head_number, self.head_dim).transpose(1, 2)
 
        # Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
 
        # Masque causal (broadcast sur batch et têtes)
        mask = self.build_causal_mask(seq_len, x.device)
        scores = scores.masked_fill(mask, float("-inf"))
 
        attention_weights = torch.softmax(scores, dim=-1)
 
        attention = torch.matmul(attention_weights, V)
 
        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, seq_len, self.embedding_dim)
 
        output = self.out(attention)
 
        return output, attention_weights
 
# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("START PROGRAM")
    print("=" * 60)

    # -------------------------------------------------
    # Check dataset
    # -------------------------------------------------

    print("Dataset path:", DATASET_PATH)
    print("Exists:", DATASET_PATH.exists())
    print()

    if not DATASET_PATH.exists():
        print("ERROR: Dataset file not found!")
        return

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------

    text = load_text(DATASET_PATH)

    print("=" * 60)
    print("RAW TEXT")
    print("=" * 60)
    print(text)
    print()

    text = clean_text(text)

    print("=" * 60)
    print("CLEAN TEXT")
    print("=" * 60)
    print(text)
    print()

    if len(text) == 0:
        print("ERROR: Dataset is empty.")
        return

    # -------------------------------------------------
    # Vocabulary
    # -------------------------------------------------

    word_to_id, id_to_word = build_vocabulary(text)

    print("=" * 60)
    print("VOCABULARY SIZE")
    print("=" * 60)
    print(len(word_to_id))
    print()

    # -------------------------------------------------
    # Encode
    # -------------------------------------------------

    token_ids = encode(text, word_to_id)

    print("=" * 60)
    print("TOKEN IDS")
    print("=" * 60)
    print(token_ids)
    print()

    print("Number of tokens:", len(token_ids))
    print()

    if len(token_ids) <= SEQUENCE_LENGTH:
        print("ERROR")
        print("The dataset is too small.")
        print(f"Need at least {SEQUENCE_LENGTH + 1} tokens.")
        return

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    dataset = TextDataset(
        token_ids,
        SEQUENCE_LENGTH
    )

    print("=" * 60)
    print("DATASET SIZE")
    print("=" * 60)
    print(len(dataset))
    print()

    if len(dataset) == 0:
        print("ERROR: Dataset contains no samples.")
        return

    # -------------------------------------------------
    # DataLoader
    # -------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # -------------------------------------------------
    # Layers
    # -------------------------------------------------

    embedding = nn.Embedding(
        num_embeddings=len(word_to_id),
        embedding_dim=EMBEDDING_DIM
    )

    positional_encoding = PositionalEncoding(
        EMBEDDING_DIM
    )

    masked_attention = MaskedMultiHeadAttention(
        EMBEDDING_DIM
    )

    masked_multi_head = MaskedMultiHeadSelfAttention(
        EMBEDDING_DIM,
        HEAD_NUMBER
    )

    masked_attention.eval()
    masked_multi_head.eval()

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------

    for batch, (inputs, targets) in enumerate(dataloader):

        print("=" * 60)
        print(f"BATCH {batch}")
        print("=" * 60)

        print("Input IDs")
        print(inputs)
        print()

        embeddings = embedding(inputs)

        print("Embedding Shape")
        print(embeddings.shape)
        print()

        embeddings = positional_encoding(embeddings)

        print("After Positional Encoding")
        print(embeddings.shape)
        print()

        # -----------------------------
        # Single Head
        # -----------------------------

        single_output, single_weights = masked_attention(
            embeddings
        )

        print("=" * 60)
        print("SINGLE HEAD MASKED ATTENTION")
        print("=" * 60)

        print("Output Shape:", single_output.shape)
        print("Attention Shape:", single_weights.shape)
        print()

        print(single_weights[0])
        print()

        # -----------------------------
        # Multi Head
        # -----------------------------

        multi_output, multi_weights = masked_multi_head(
            embeddings
        )

        print("=" * 60)
        print("MASKED MULTI HEAD")
        print("=" * 60)

        print("Output Shape:", multi_output.shape)
        print("Attention Shape:", multi_weights.shape)
        print()

        print("Head 0")
        print(multi_weights[0, 0])
        print()

        # -----------------------------
        # Tokens
        # -----------------------------

        print("=" * 60)
        print("TOKEN REPRESENTATIONS")
        print("=" * 60)

        first_sentence = inputs[0]

        for i, token in enumerate(first_sentence):

            idx = token.item()

            print(
                f"{id_to_word[idx]:12} -> {idx:3d} -> {multi_output[0, i, :5]}"
            )

        break

    print("=" * 60)
    print("PROGRAM FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()