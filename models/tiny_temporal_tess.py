"""Tiny temporal-token Transformer models for 5-in-5-out smoke tests."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class TinyTemporalForecaster(nn.Module):
    """Numeric-only Transformer over scalar time-step tokens."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pooling: str = "flatten",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pooling = pooling
        if pooling not in {"flatten", "mean"}:
            raise ValueError("pooling must be 'flatten' or 'mean'")

        self.value_projection = nn.Linear(1, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        head_dim = seq_len * d_model if pooling == "flatten" else d_model
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, L, 1], got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")
        tokens = self.value_projection(x) + self.positional_embedding
        encoded = self.encoder(tokens)
        if self.pooling == "flatten":
            features = encoded.flatten(start_dim=1)
        else:
            features = encoded.mean(dim=1)
        return self.head(features).unsqueeze(-1)


class TinyTemporalTESS(nn.Module):
    """Tiny Transformer using primitive tokens plus numeric time-step tokens."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        primitive_vocab_sizes: Sequence[int] = (6, 4, 6, 4),
        pooling: str = "flatten",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pooling = pooling
        self.primitive_vocab_sizes = tuple(primitive_vocab_sizes)
        if len(self.primitive_vocab_sizes) != 4:
            raise ValueError("Expected four primitive vocab sizes.")
        if pooling not in {"flatten", "numeric_mean"}:
            raise ValueError("pooling must be 'flatten' or 'numeric_mean'")

        self.value_projection = nn.Linear(1, d_model)
        self.numeric_positional_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.primitive_embeddings = nn.ModuleList(
            nn.Embedding(vocab_size, d_model) for vocab_size in self.primitive_vocab_sizes
        )
        self.primitive_type_embedding = nn.Parameter(torch.zeros(1, 4, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        total_tokens = seq_len + 4
        head_dim = total_tokens * d_model if pooling == "flatten" else d_model
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, pred_len),
        )

    def forward(
        self,
        x: torch.Tensor,
        primitive_ids: torch.Tensor,
        primitive_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, L, 1], got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")
        if primitive_ids.ndim != 2 or primitive_ids.shape[1] != 4:
            raise ValueError(f"Expected primitive_ids [B, 4], got {tuple(primitive_ids.shape)}")

        primitive_tokens = []
        for idx, embedding in enumerate(self.primitive_embeddings):
            primitive_tokens.append(embedding(primitive_ids[:, idx]))
        primitive_tokens_tensor = torch.stack(primitive_tokens, dim=1)
        primitive_tokens_tensor = primitive_tokens_tensor + self.primitive_type_embedding

        numeric_tokens = self.value_projection(x) + self.numeric_positional_embedding
        tokens = torch.cat([primitive_tokens_tensor, numeric_tokens], dim=1)
        encoded = self.encoder(tokens)
        if self.pooling == "flatten":
            features = encoded.flatten(start_dim=1)
        else:
            features = encoded[:, 4:, :].mean(dim=1)
        return self.head(features).unsqueeze(-1)
