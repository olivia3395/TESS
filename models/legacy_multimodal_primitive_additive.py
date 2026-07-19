"""Late-additive primitive variant of the legacy multimodal baseline.

This model keeps the legacy numeric temporal encoder and primitive embedding
path, but decodes each branch separately and adds their predictions in
normalized forecast space.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.fft as fft

from .legacy_timesnet import DataEmbedding


class SelfAttention(nn.Module):
    def __init__(self, num_heads: int, in_dim: int, hid_dim: int, dropout: float) -> None:
        super().__init__()
        if hid_dim % num_heads != 0:
            raise ValueError("hid_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hid_dim // num_heads
        self.hid_dim = hid_dim
        self.query = nn.Linear(in_dim, hid_dim)
        self.key = nn.Linear(in_dim, hid_dim)
        self.value = nn.Linear(in_dim, hid_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hid_dim, hid_dim)
        self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-12)
        self.out_dropout = nn.Dropout(dropout)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        query = self._transpose_for_scores(self.query(input_tensor))
        key = self._transpose_for_scores(self.key(input_tensor))
        value = self._transpose_for_scores(self.value(input_tensor))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        probs = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(probs, value).permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:-2], self.hid_dim)
        hidden = self.out_dropout(self.dense(context))
        return self.layer_norm(hidden + input_tensor)


class TempEncoder(nn.Module):
    """Legacy temporal encoder: causal conv environment branch plus time/frequency entity branch."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernels: Sequence[int],
        length: int,
        hidden_dims: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        del depth  # The legacy constructor accepts depth but this block does not use it.
        component_dims = output_dims // 2
        self.hidden_dims = hidden_dims
        self.kernels = tuple(kernels)
        self.env_encoder = nn.ModuleList(
            nn.Conv1d(input_dims, component_dims, kernel, padding=kernel - 1)
            for kernel in self.kernels
        )
        self.entity_time = SelfAttention(
            num_heads=4,
            in_dim=input_dims,
            hid_dim=hidden_dims,
            dropout=dropout,
        )
        self.length = length
        self.num_freqs = (self.length // 2) + 1
        self.entity_freq_weight = nn.Parameter(
            torch.empty((self.num_freqs, hidden_dims, hidden_dims), dtype=torch.cfloat)
        )
        self.entity_freq_bias = nn.Parameter(
            torch.empty((self.num_freqs, hidden_dims), dtype=torch.cfloat)
        )
        self.reset_parameters()
        self.entity_dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.entity_freq_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.entity_freq_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.entity_freq_bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_input = x.transpose(1, 2)
        env_chunks = []
        for kernel, conv in zip(self.kernels, self.env_encoder):
            out = conv(conv_input)
            if kernel != 1:
                out = out[..., : -(kernel - 1)]
            env_chunks.append(out.transpose(1, 2))
        env_rep = torch.stack(env_chunks, dim=0).mean(0)

        entity_time = self.entity_time(x)
        input_freq = fft.rfft(x, dim=1)
        output_freq = torch.zeros(
            x.size(0),
            x.size(1) // 2 + 1,
            self.hidden_dims,
            device=x.device,
            dtype=torch.cfloat,
        )
        output_freq[:, : self.num_freqs] = (
            torch.einsum(
                "bti,tio->bto",
                input_freq[:, : self.num_freqs],
                self.entity_freq_weight,
            )
            + self.entity_freq_bias
        )
        entity_freq = fft.irfft(output_freq, n=x.size(1), dim=1)
        entity_rep = self.entity_dropout(entity_time + entity_freq)
        return env_rep, entity_rep


class LegacyMultimodalPrimitiveAdditive(nn.Module):
    """Late-additive primitive version of legacy `MultiModal_Baseline`."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 256,
        primitive_emb_dim: int = 32,
        primitive_vocab_sizes: Sequence[int] = (6, 4, 6, 4),
        depth: int = 10,
        dropout: float = 0.3,
        dropout2: float = 0.3,
        primitive_decoder_hidden: int = 256,
        embedding_dropout: float = 0.1,
        text_delta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.text_delta_scale = float(text_delta_scale)
        self.primitive_vocab_sizes = tuple(primitive_vocab_sizes)
        if len(self.primitive_vocab_sizes) != 4:
            raise ValueError("Expected four primitive vocab sizes.")

        self.enc_embedding = DataEmbedding(c_in=1, d_model=d_model, dropout=embedding_dropout)
        kernels = [2**idx for idx in range(int(math.log2(seq_len // 2)))]
        if not kernels:
            kernels = [1]
        self.temporal = TempEncoder(
            input_dims=d_model,
            output_dims=d_model * 2,
            kernels=kernels,
            length=seq_len,
            hidden_dims=d_model,
            depth=depth,
            dropout=dropout,
        )
        self.mlp_flatten = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.numerical_decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, pred_len),
        )

        self.primitive_embeddings = nn.ModuleList(
            nn.Embedding(vocab_size, primitive_emb_dim)
            for vocab_size in self.primitive_vocab_sizes
        )
        self.dynamic_fc = nn.Sequential(
            nn.Linear(4 * primitive_emb_dim, d_model),
            nn.PReLU(),
            nn.Dropout(dropout2),
        )
        self.primitive_decoder = nn.Sequential(
            nn.Linear(d_model, primitive_decoder_hidden),
            nn.PReLU(),
            nn.Dropout(dropout2),
            nn.Linear(primitive_decoder_hidden, pred_len),
        )

    def _primitive_feature(
        self,
        primitive_ids: torch.Tensor,
        primitive_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        chunks = []
        for idx, embedding in enumerate(self.primitive_embeddings):
            chunk = embedding(primitive_ids[:, idx])
            if primitive_mask is not None:
                chunk = chunk * primitive_mask[:, idx].to(dtype=chunk.dtype).unsqueeze(-1)
            chunks.append(chunk)
        return torch.cat(chunks, dim=-1)

    def _primitive_embeddings_3d(
        self,
        primitive_ids: torch.Tensor,
        primitive_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        chunks = []
        for idx, embedding in enumerate(self.primitive_embeddings):
            chunk = embedding(primitive_ids[:, idx])
            if primitive_mask is not None:
                chunk = chunk * primitive_mask[:, idx].to(dtype=chunk.dtype).unsqueeze(-1)
            chunks.append(chunk)
        return torch.stack(chunks, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        primitive_ids: torch.Tensor,
        primitive_mask: torch.Tensor | None = None,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, L, 1], got {tuple(x.shape)}")
        if primitive_ids.ndim != 2 or primitive_ids.shape[1] != 4:
            raise ValueError(f"Expected primitive_ids [B, 4], got {tuple(primitive_ids.shape)}")
        if primitive_mask is not None and (
            primitive_mask.ndim != 2 or primitive_mask.shape[1] != 4
        ):
            raise ValueError(f"Expected primitive_mask [B, 4], got {tuple(primitive_mask.shape)}")

        means = x.mean(1, keepdim=True).detach()
        x_norm = x - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev

        enc_out = self.enc_embedding(x_norm)
        _, h_ts = self.temporal(enc_out)
        h_ts = self.mlp_flatten(h_ts.reshape(x.shape[0], -1))
        y_num_norm = self.numerical_decoder(h_ts)

        primitive_feature = self._primitive_feature(primitive_ids, primitive_mask)
        h_primitive = self.dynamic_fc(primitive_feature)
        y_primitive_delta_norm = self.primitive_decoder(h_primitive)

        y_hat_norm = y_num_norm + self.text_delta_scale * y_primitive_delta_norm
        y_hat = y_hat_norm.unsqueeze(-1) * stdev + means

        if not return_components:
            return y_hat

        y_num = y_num_norm.unsqueeze(-1) * stdev + means
        y_primitive_delta = (
            self.text_delta_scale * y_primitive_delta_norm
        ).unsqueeze(-1) * stdev
        return {
            "y_hat": y_hat,
            "y_num": y_num,
            "y_primitive_delta": y_primitive_delta,
            "y_num_norm": y_num_norm,
            "y_primitive_delta_norm": y_primitive_delta_norm,
        }


class LegacyMultimodalPrimitiveAdditiveGate(LegacyMultimodalPrimitiveAdditive):
    """Late-additive primitive variant with a dynamic residual scale gate."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 256,
        primitive_emb_dim: int = 32,
        primitive_vocab_sizes: Sequence[int] = (6, 4, 6, 4),
        depth: int = 10,
        dropout: float = 0.3,
        dropout2: float = 0.3,
        primitive_decoder_hidden: int = 256,
        embedding_dropout: float = 0.1,
        margin_emb_dim: int = 8,
        time_gate_dim: int = 16,
        text_delta_scale: float = 1.0,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            primitive_emb_dim=primitive_emb_dim,
            primitive_vocab_sizes=primitive_vocab_sizes,
            depth=depth,
            dropout=dropout,
            dropout2=dropout2,
            primitive_decoder_hidden=primitive_decoder_hidden,
            embedding_dropout=embedding_dropout,
            text_delta_scale=text_delta_scale,
        )
        self.margin_projection = nn.Sequential(
            nn.Linear(1, margin_emb_dim),
            nn.PReLU(),
        )
        self.time_gate_projection = nn.Sequential(
            nn.Linear(d_model, time_gate_dim),
            nn.PReLU(),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(primitive_emb_dim + margin_emb_dim + time_gate_dim, primitive_emb_dim),
            nn.PReLU(),
            nn.Linear(primitive_emb_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        primitive_ids: torch.Tensor,
        primitive_mask: torch.Tensor | None = None,
        primitive_margins: torch.Tensor | None = None,
        *,
        return_components: bool = False,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, L, 1], got {tuple(x.shape)}")
        if primitive_ids.ndim != 2 or primitive_ids.shape[1] != 4:
            raise ValueError(f"Expected primitive_ids [B, 4], got {tuple(primitive_ids.shape)}")
        if primitive_mask is not None and (
            primitive_mask.ndim != 2 or primitive_mask.shape[1] != 4
        ):
            raise ValueError(f"Expected primitive_mask [B, 4], got {tuple(primitive_mask.shape)}")
        if primitive_margins is None:
            primitive_margins = torch.zeros(
                primitive_ids.shape,
                dtype=x.dtype,
                device=x.device,
            )
        primitive_margins = primitive_margins.to(device=x.device, dtype=x.dtype)

        means = x.mean(1, keepdim=True).detach()
        x_norm = x - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev

        enc_out = self.enc_embedding(x_norm)
        _, h_ts = self.temporal(enc_out)
        h_ts = self.mlp_flatten(h_ts.reshape(x.shape[0], -1))
        y_num_norm = self.numerical_decoder(h_ts)

        primitive_emb = self._primitive_embeddings_3d(primitive_ids, primitive_mask)
        primitive_feature = primitive_emb.reshape(x.shape[0], -1)
        h_primitive = self.dynamic_fc(primitive_feature)
        y_primitive_delta_norm = self.primitive_decoder(h_primitive)

        margin_emb = self.margin_projection(primitive_margins.unsqueeze(-1))
        time_context = self.time_gate_projection(h_ts).unsqueeze(1).expand(-1, 4, -1)
        gate_input = torch.cat([primitive_emb, margin_emb, time_context], dim=-1)
        gate_logits = self.gate_head(gate_input).squeeze(-1)
        if primitive_mask is not None:
            gate_logits = gate_logits.masked_fill(~primitive_mask, -20.0)
        gate_weights = torch.sigmoid(gate_logits)
        if primitive_mask is None:
            dynamic_scale = gate_weights.mean(dim=1, keepdim=True)
        else:
            valid = primitive_mask.to(device=x.device, dtype=gate_weights.dtype)
            dynamic_scale = (gate_weights * valid).sum(dim=1, keepdim=True)
            dynamic_scale = dynamic_scale / valid.sum(dim=1, keepdim=True).clamp_min(1.0)

        y_hat_norm = y_num_norm + self.text_delta_scale * dynamic_scale * y_primitive_delta_norm
        y_hat = y_hat_norm.unsqueeze(-1) * stdev + means
        output = {
            "y_hat": y_hat,
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
            "dynamic_scale": dynamic_scale.squeeze(-1),
        }
        if not return_components:
            return output

        y_num = y_num_norm.unsqueeze(-1) * stdev + means
        y_primitive_delta = (
            self.text_delta_scale * dynamic_scale * y_primitive_delta_norm
        ).unsqueeze(-1) * stdev
        output.update(
            {
                "y_num": y_num,
                "y_primitive_delta": y_primitive_delta,
                "y_num_norm": y_num_norm,
                "y_primitive_delta_norm": y_primitive_delta_norm,
            }
        )
        return output


class LegacyMultimodalPrimitiveAdditiveSoft(LegacyMultimodalPrimitiveAdditive):
    """Late-additive primitive model with optional sampled-probability soft embeddings."""

    uses_soft_primitive_probs = True

    def _primitive_feature_soft(
        self,
        primitive_ids: torch.Tensor,
        primitive_mask: torch.Tensor | None,
        primitive_probs: torch.Tensor | None,
        primitive_prob_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if primitive_probs is None:
            return self._primitive_feature(primitive_ids, primitive_mask)
        chunks = []
        for idx, embedding in enumerate(self.primitive_embeddings):
            vocab_size = embedding.num_embeddings
            probs = primitive_probs[:, idx, :vocab_size].to(
                device=embedding.weight.device,
                dtype=embedding.weight.dtype,
            )
            if primitive_prob_mask is not None:
                probs = probs * primitive_prob_mask[:, idx, :vocab_size].to(
                    device=embedding.weight.device,
                    dtype=embedding.weight.dtype,
                )
            # Exclude the UNK row from sampled label distributions unless a future cache
            # explicitly assigns it nonzero probability.
            probs[:, -1] = 0.0
            denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            chunk = torch.matmul(probs / denom, embedding.weight)
            if primitive_mask is not None:
                chunk = chunk * primitive_mask[:, idx].to(dtype=chunk.dtype).unsqueeze(-1)
            chunks.append(chunk)
        return torch.cat(chunks, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        primitive_ids: torch.Tensor,
        primitive_mask: torch.Tensor | None = None,
        primitive_probs: torch.Tensor | None = None,
        primitive_prob_mask: torch.Tensor | None = None,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, L, 1], got {tuple(x.shape)}")
        if primitive_ids.ndim != 2 or primitive_ids.shape[1] != 4:
            raise ValueError(f"Expected primitive_ids [B, 4], got {tuple(primitive_ids.shape)}")
        if primitive_mask is not None and (
            primitive_mask.ndim != 2 or primitive_mask.shape[1] != 4
        ):
            raise ValueError(f"Expected primitive_mask [B, 4], got {tuple(primitive_mask.shape)}")
        if primitive_probs is not None and (
            primitive_probs.ndim != 3 or primitive_probs.shape[1] != 4
        ):
            raise ValueError(f"Expected primitive_probs [B, 4, V], got {tuple(primitive_probs.shape)}")

        means = x.mean(1, keepdim=True).detach()
        x_norm = x - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev

        enc_out = self.enc_embedding(x_norm)
        _, h_ts = self.temporal(enc_out)
        h_ts = self.mlp_flatten(h_ts.reshape(x.shape[0], -1))
        y_num_norm = self.numerical_decoder(h_ts)

        primitive_feature = self._primitive_feature_soft(
            primitive_ids,
            primitive_mask,
            primitive_probs,
            primitive_prob_mask,
        )
        h_primitive = self.dynamic_fc(primitive_feature)
        y_primitive_delta_norm = self.primitive_decoder(h_primitive)

        y_hat_norm = y_num_norm + self.text_delta_scale * y_primitive_delta_norm
        y_hat = y_hat_norm.unsqueeze(-1) * stdev + means

        if not return_components:
            return y_hat

        y_num = y_num_norm.unsqueeze(-1) * stdev + means
        y_primitive_delta = (
            self.text_delta_scale * y_primitive_delta_norm
        ).unsqueeze(-1) * stdev
        return {
            "y_hat": y_hat,
            "y_num": y_num,
            "y_primitive_delta": y_primitive_delta,
            "y_num_norm": y_num_norm,
            "y_primitive_delta_norm": y_primitive_delta_norm,
        }
