"""Simple trainable forecasting models for pipeline validation.

These models are intentionally small. They validate data loading, primitive
embedding plumbing, checkpointing, and evaluation before implementing the final
paper-faithful architecture.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .legacy_multimodal_primitive import (
    LegacyMultimodalPrimitive,
    LegacyMultimodalPrimitiveDeltaGate,
    LegacyMultimodalPrimitiveGate,
)
from .legacy_multimodal_primitive_additive import (
    LegacyMultimodalPrimitiveAdditive,
    LegacyMultimodalPrimitiveAdditiveGate,
    LegacyMultimodalPrimitiveAdditiveSoft,
)
from .legacy_timesnet import LegacyTimesNet
from .tiny_temporal_tess import TinyTemporalForecaster, TinyTemporalTESS


class NumericMLPForecaster(nn.Module):
    """Numerical-only MLP baseline using x [B, L, 1]."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, L, 1], got {tuple(x.shape)}")
        h = self.encoder(x.squeeze(-1))
        return self.head(h).unsqueeze(-1)


class TESSNoGateForecaster(nn.Module):
    """Minimal no-gate TESS model using text or oracle primitive IDs."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        primitive_emb_dim: int = 32,
        primitive_vocab_sizes: Sequence[int] = (6, 4, 6, 4),
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.primitive_vocab_sizes = tuple(primitive_vocab_sizes)
        if len(self.primitive_vocab_sizes) != 4:
            raise ValueError("Expected four primitive vocab sizes.")

        self.numeric_encoder = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.primitive_embeddings = nn.ModuleList(
            nn.Embedding(vocab_size, primitive_emb_dim)
            for vocab_size in self.primitive_vocab_sizes
        )
        fused_dim = hidden_dim + 4 * primitive_emb_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(
        self,
        x: torch.Tensor,
        primitive_ids: torch.Tensor,
        primitive_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, L, 1], got {tuple(x.shape)}")
        if primitive_ids.ndim != 2 or primitive_ids.shape[1] != 4:
            raise ValueError(f"Expected primitive_ids [B, 4], got {tuple(primitive_ids.shape)}")

        h_num = self.numeric_encoder(x.squeeze(-1))
        primitive_chunks = []
        for idx, embedding in enumerate(self.primitive_embeddings):
            emb = embedding(primitive_ids[:, idx])
            if primitive_mask is not None:
                emb = emb * primitive_mask[:, idx].to(dtype=emb.dtype).unsqueeze(-1)
            primitive_chunks.append(emb)
        h_primitive = torch.cat(primitive_chunks, dim=-1)
        fused = torch.cat([h_num, h_primitive], dim=-1)
        return self.head(fused).unsqueeze(-1)


def build_model(
    model_name: str,
    seq_len: int,
    pred_len: int,
    **kwargs,
) -> nn.Module:
    """Build a supported smoke-test forecasting model."""

    if model_name == "numeric_mlp":
        return NumericMLPForecaster(seq_len=seq_len, pred_len=pred_len, **kwargs)
    if model_name == "tess_nogate":
        return TESSNoGateForecaster(seq_len=seq_len, pred_len=pred_len, **kwargs)
    if model_name == "tiny_temporal":
        return TinyTemporalForecaster(seq_len=seq_len, pred_len=pred_len, **kwargs)
    if model_name == "tiny_temporal_tess":
        return TinyTemporalTESS(seq_len=seq_len, pred_len=pred_len, **kwargs)
    if model_name == "legacy_timesnet":
        return LegacyTimesNet(seq_len=seq_len, pred_len=pred_len, **kwargs)
    if model_name == "legacy_multimodal_primitive":
        return LegacyMultimodalPrimitive(seq_len=seq_len, pred_len=pred_len, **kwargs)
    if model_name == "legacy_multimodal_primitive_additive":
        return LegacyMultimodalPrimitiveAdditive(seq_len=seq_len, pred_len=pred_len, **kwargs)
    if model_name == "legacy_multimodal_primitive_additive_soft":
        return LegacyMultimodalPrimitiveAdditiveSoft(seq_len=seq_len, pred_len=pred_len, **kwargs)
    if model_name == "legacy_multimodal_primitive_additive_gate":
        return LegacyMultimodalPrimitiveAdditiveGate(seq_len=seq_len, pred_len=pred_len, **kwargs)
    if model_name == "legacy_multimodal_primitive_gate":
        return LegacyMultimodalPrimitiveGate(seq_len=seq_len, pred_len=pred_len, **kwargs)
    if model_name == "legacy_multimodal_primitive_delta_gate":
        return LegacyMultimodalPrimitiveDeltaGate(seq_len=seq_len, pred_len=pred_len, **kwargs)
    raise ValueError(
        "Supported model names: numeric_mlp, tess_nogate, tiny_temporal, "
        "tiny_temporal_tess, legacy_timesnet, legacy_multimodal_primitive, "
        "legacy_multimodal_primitive_additive, legacy_multimodal_primitive_additive_soft, "
        "legacy_multimodal_primitive_additive_gate, legacy_multimodal_primitive_gate, "
        "legacy_multimodal_primitive_delta_gate"
    )
