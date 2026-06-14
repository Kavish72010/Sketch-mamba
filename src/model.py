"""SketchMamba and matched-parameter baseline backbones.

All models share the same I/O contract:
    input  : float tensor (B, T, 5)  stroke-5 features
    output : dict with
        h          : (B, T, D)   per-step hidden state (causal)
        cls_logits : (B, T, C)   per-step class logits
        gmm        : (B, T, 6M)  raw GMM parameters  (mixture, μx, μy, σx, σy, ρ)
        pen_logits : (B, T, 3)   per-step pen-state logits

Causal (Mamba, GRU, LSTM, causal-Transformer, causal 1D-CNN) backbones make
the progressive-classification curve fall out for free: at any prefix
length t the user reads ``cls_logits[:, t-1]``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).to(x.dtype) * self.weight


class SketchHeads(nn.Module):
    """Per-step classification + GMM + pen heads."""

    def __init__(self, d_model: int, num_classes: int, gmm_components: int) -> None:
        super().__init__()
        self.cls = nn.Linear(d_model, num_classes)
        self.gmm = nn.Linear(d_model, 6 * gmm_components)
        self.pen = nn.Linear(d_model, 3)

    def forward(self, h: torch.Tensor) -> dict:
        return {
            "h": h,
            "cls_logits": self.cls(h),
            "gmm": self.gmm(h),
            "pen_logits": self.pen(h),
        }


# --------------------------------------------------------------------------- #
# SketchMamba
# --------------------------------------------------------------------------- #
def _resolve_mamba_cls():
    """Return the Mamba implementation to use.

    Prefers the official CUDA-fused `mamba_ssm.Mamba`; falls back to a pure
    PyTorch reference (`MambaCPU`) on platforms where `mamba-ssm` is not
    installable (Windows, CPU-only, macOS).  Both classes share the same
    parameter names and shapes, so the same checkpoint loads in either.
    """
    try:
        from mamba_ssm import Mamba
        return Mamba
    except Exception:  # noqa: BLE001  (covers ImportError + CUDA-load failures)
        from src.mamba_cpu import MambaCPU
        return MambaCPU


class _MambaResBlock(nn.Module):
    def __init__(
        self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float
    ) -> None:
        super().__init__()
        mamba_cls = _resolve_mamba_cls()
        self.norm = RMSNorm(d_model)
        self.mamba = mamba_cls(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mamba(self.norm(x)))


class SketchMamba(nn.Module):
    def __init__(
        self,
        num_classes: int = 58,
        d_model: int = 192,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        gmm_components: int = 20,
        dropout: float = 0.1,
        in_dim: int = 5,
        **_: object,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                _MambaResBlock(d_model, d_state, d_conv, expand, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.heads = SketchHeads(d_model, num_classes, gmm_components)
        self.gmm_components = gmm_components
        self.num_classes = num_classes
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.final_norm(h)
        return self.heads(h)


# --------------------------------------------------------------------------- #
# GRU baseline (causal)
# --------------------------------------------------------------------------- #
class SketchGRU(nn.Module):
    def __init__(
        self,
        num_classes: int = 58,
        d_model: int = 192,
        n_layers: int = 3,
        gmm_components: int = 20,
        dropout: float = 0.1,
        in_dim: int = 5,
        **_: object,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.rnn = nn.GRU(
            d_model,
            d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.heads = SketchHeads(d_model, num_classes, gmm_components)
        self.gmm_components = gmm_components
        self.num_classes = num_classes
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        h, _ = self.rnn(self.embed(x))
        return self.heads(self.final_norm(h))


# --------------------------------------------------------------------------- #
# LSTM baseline (causal, Sketch-RNN-style)
# --------------------------------------------------------------------------- #
class SketchLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 58,
        d_model: int = 192,
        n_layers: int = 3,
        gmm_components: int = 20,
        dropout: float = 0.1,
        in_dim: int = 5,
        **_: object,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.rnn = nn.LSTM(
            d_model,
            d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.heads = SketchHeads(d_model, num_classes, gmm_components)
        self.gmm_components = gmm_components
        self.num_classes = num_classes
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        h, _ = self.rnn(self.embed(x))
        return self.heads(self.final_norm(h))


# --------------------------------------------------------------------------- #
# Causal Transformer baseline
# --------------------------------------------------------------------------- #
class SketchTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 58,
        d_model: int = 192,
        n_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 4 * 192,
        gmm_components: int = 20,
        dropout: float = 0.1,
        max_len: int = 256,
        in_dim: int = 5,
        **_: object,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.heads = SketchHeads(d_model, num_classes, gmm_components)
        self.gmm_components = gmm_components
        self.num_classes = num_classes
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len={self.max_len}")
        h = self.embed(x) + self.pos[:, :T]
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # True where padding
        h = self.encoder(h, mask=causal, src_key_padding_mask=key_padding_mask)
        return self.heads(self.final_norm(h))


# --------------------------------------------------------------------------- #
# Causal 1D-CNN baseline
# --------------------------------------------------------------------------- #
class _CausalConv1d(nn.Module):
    def __init__(self, d: int, kernel: int = 5, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(d, d, kernel_size=kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class SketchCNN1D(nn.Module):
    def __init__(
        self,
        num_classes: int = 58,
        d_model: int = 192,
        n_layers: int = 6,
        gmm_components: int = 20,
        dropout: float = 0.1,
        kernel: int = 5,
        in_dim: int = 5,
        **_: object,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    _CausalConv1d(d_model, kernel=kernel, dilation=2 ** (i % 4)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for i in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.heads = SketchHeads(d_model, num_classes, gmm_components)
        self.gmm_components = gmm_components
        self.num_classes = num_classes
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        h = self.embed(x)  # (B, T, D)
        for norm, layer in zip(self.norms, self.layers):
            t = norm(h).transpose(1, 2)  # (B, D, T)
            t = layer(t).transpose(1, 2)  # (B, T, D)
            h = h + t
        return self.heads(self.final_norm(h))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "sketchmamba": SketchMamba,
    "gru": SketchGRU,
    "lstm": SketchLSTM,
    "transformer": SketchTransformer,
    "cnn1d": SketchCNN1D,
}


def build_model(model_cfg: dict) -> nn.Module:
    cfg = dict(model_cfg)
    mtype = cfg.pop("type", "sketchmamba")
    if mtype not in _REGISTRY:
        raise ValueError(
            f"Unknown model type '{mtype}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[mtype](**cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# GMM parameter parsing (used by losses + sampling)
# --------------------------------------------------------------------------- #
@dataclass
class GMMParams:
    pi: torch.Tensor      # (..., M)  log-mixture weights (log-softmax)
    mu_x: torch.Tensor    # (..., M)
    mu_y: torch.Tensor    # (..., M)
    sigma_x: torch.Tensor # (..., M)  positive
    sigma_y: torch.Tensor # (..., M)  positive
    rho: torch.Tensor     # (..., M)  in (-1, 1)


def parse_gmm(raw: torch.Tensor, n_components: int) -> GMMParams:
    """raw : (..., 6M)  ->  GMMParams with stable activations."""
    M = n_components
    pi, mu_x, mu_y, log_sx, log_sy, raw_rho = torch.split(raw, M, dim=-1)
    log_pi = F.log_softmax(pi, dim=-1)
    sigma_x = F.softplus(log_sx) + 1e-4
    sigma_y = F.softplus(log_sy) + 1e-4
    rho = torch.tanh(raw_rho).clamp(-0.99, 0.99)
    return GMMParams(log_pi, mu_x, mu_y, sigma_x, sigma_y, rho)
