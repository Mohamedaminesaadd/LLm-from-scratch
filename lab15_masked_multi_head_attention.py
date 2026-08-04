import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lab06_positional_encoding import (
    PositionalEncoding,
    TextDataset,
    build_vocabulary,
    clean_text,
    encode,
    load_text,
)


DATASET_PATH = Path("datasets/sample.txt")

SEQUENCE_LENGTH = 5
BATCH_SIZE = 2
EMBEDDING_DIM = 100
HEAD_NUMBER = 4


class MaskedSelfAttention(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.embedding_dim
        )

        seq_len = x.size(1)

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention_weights,
            V
        )

        return output, attention_weights


class MaskedMultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        head_number
    ):
        super().__init__()

        assert embedding_dim % head_number == 0

        self.embedding_dim = embedding_dim
        self.head_number = head_number
        self.head_dim = embedding_dim // head_number

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.out = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.head_number,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len,
            self.head_number,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.head_number,
            self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        attention = torch.matmul(
            attention_weights,
            V
        )

        attention = attention.transpose(
            1,
            2
        ).contiguous()

        attention = attention.view(
            batch_size,
            seq_len,
            self.embedding_dim
        )

        output = self.out(attention)

        return output, attention_weights


def main():

    text = load_text(DATASET_PATH)
    text = clean_text(text)

    word_to_id, id_to_word = build_vocabulary(text)

    token_ids = encode(
        text,
        word_to_id
    )

    dataset = TextDataset(
        token_ids,
        SEQUENCE_LENGTH
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    embedding = nn.Embedding(
        num_embeddings=len(word_to_id),
        embedding_dim=EMBEDDING_DIM
    )

    positional_encoding = PositionalEncoding(
        EMBEDDING_DIM
    )

    masked_attention = MaskedSelfAttention(
        EMBEDDING_DIM
    )

    masked_multi_head_attention = MaskedMultiHeadSelfAttention(
        EMBEDDING_DIM,
        HEAD_NUMBER
    )

    masked_attention.eval()
    masked_multi_head_attention.eval()

    for inputs, targets in dataloader:

        print("=" * 60)
        print("INPUT TOKEN IDS")
        print("=" * 60)
        print(inputs)
        print()

        embeddings = embedding(inputs)

        embeddings = positional_encoding(
            embeddings
        )

        print("=" * 60)
        print("EMBEDDING SHAPE")
        print("=" * 60)
        print(embeddings.shape)
        print()

        single_output, single_weights = masked_attention(
            embeddings
        )

        print("=" * 60)
        print("SINGLE-HEAD MASKED SELF-ATTENTION")
        print("=" * 60)

        print(
            "Output shape:",
            single_output.shape
        )

        print(
            "Attention shape:",
            single_weights.shape
        )

        print()

        print("Attention matrix:")

        print(
            single_weights[0]
        )

        print()

        multi_output, multi_weights = (
            masked_multi_head_attention(
                embeddings
            )
        )

        print("=" * 60)
        print("MASKED MULTI-HEAD SELF-ATTENTION")
        print("=" * 60)

        print(
            "Output shape:",
            multi_output.shape
        )

        print(
            "Attention shape:",
            multi_weights.shape
        )

        print()

        for head in range(HEAD_NUMBER):

            print(f"Head {head}:")

            print(
                multi_weights[
                    0,
                    head
                ]
            )

            print()

        print("=" * 60)
        print("TOKENS AND OUTPUT REPRESENTATIONS")
        print("=" * 60)

        first_sequence = inputs[0]

        for position, token_id in enumerate(
            first_sequence
        ):

            idx = token_id.item()

            word = id_to_word[idx]

            vector = multi_output[
                0,
                position,
                :5
            ]

            print(
                f"{word:15} -> "
                f"{idx:3d} -> "
                f"{vector}"
            )

        break


if __name__ == "__main__":
    main()