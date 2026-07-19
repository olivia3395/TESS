"""Legacy-compatible numeric TimesNet for RC2 experiments.

This ports the legacy `model_trainer.models.timesnet.TimesNet` architecture
without importing the legacy training package. It preserves the internal
Non-stationary-Transformer style instance normalization used by the legacy
model, while the dataset-level scale is controlled separately by
`experiments.train_tess_basic --scale`.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding(x).to(x.device))


class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * idx + 1, padding=idx)
            for idx in range(num_kernels)
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([kernel(x) for kernel in self.kernels], dim=-1).mean(-1)


def fft_for_period(x: torch.Tensor, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = xf.abs().mean(0).mean(-1)
    frequency_list[0] = 0
    effective_k = min(k, int(frequency_list.numel()))
    _, top_list = torch.topk(frequency_list, effective_k)
    period = x.shape[1] // top_list.detach().cpu().numpy()
    return period, xf.abs().mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, d_model: int, top_k: int, num_kernels: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_model, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_model, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels = x.size()
        period_list, period_weight = fft_for_period(x, self.k)

        results = []
        total_len = self.seq_len + self.pred_len
        for period in period_list:
            if total_len % period != 0:
                length = ((total_len // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], length - total_len, x.shape[2]],
                    device=x.device,
                    dtype=x.dtype,
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = total_len
                out = x
            out = out.reshape(batch, length // period, period, channels).permute(0, 3, 1, 2)
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(batch, -1, channels)
            results.append(out[:, :total_len, :])

        stacked = torch.stack(results, dim=-1)
        weights = F.softmax(period_weight, dim=1)
        weights = weights.unsqueeze(1).unsqueeze(1).repeat(1, time_steps, channels, 1)
        return torch.sum(stacked * weights, -1) + x


class LegacyTimesNet(nn.Module):
    """Numeric-only TimesNet with legacy FNSPID defaults."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int = 512,
        e_layers: int = 2,
        top_k: int = 5,
        num_kernels: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.layers = nn.ModuleList(
            TimesBlock(seq_len, pred_len, d_model, top_k, num_kernels)
            for _ in range(e_layers)
        )
        self.enc_embedding = DataEmbedding(c_in=1, d_model=d_model, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.predict_linear = nn.Linear(seq_len, pred_len + seq_len)
        self.projection = nn.Linear(d_model, 1, bias=True)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_out = self.enc_embedding(x_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for layer in self.layers:
            enc_out = self.layer_norm(layer(enc_out))
        dec_out = self.projection(enc_out)

        repeat_shape = (1, self.pred_len + self.seq_len, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(*repeat_shape)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(*repeat_shape)
        return dec_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError(f"Expected x [B, L, 1], got {tuple(x.shape)}")
        return self.forecast(x)[:, -self.pred_len :, :]
