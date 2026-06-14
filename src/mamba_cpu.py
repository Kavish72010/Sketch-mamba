"""Pure-PyTorch reference implementation of a Mamba block.

This is a *parameter-compatible* fallback for `mamba_ssm.Mamba`: every
sub-module name and every parameter shape matches the official CUDA
implementation, so a checkpoint trained with `mamba-ssm` loads here with
`strict=True`.  The selective-scan recurrence is unrolled in a Python
for-loop over time, so it is dramatically slower than the fused CUDA
kernel — adequate for single-sketch interactive inference (T ≤ 200,
~6 layers) but unsuitable for training or batch evaluation.

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces", 2023.  This implementation follows the
reference module in `mamba_ssm/modules/mamba_simple.py`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaCPU(nn.Module):
    """CPU/Windows-friendly drop-in for `mamba_ssm.Mamba`.

    Constructor signature is a strict subset of the upstream module.
    Parameter layout (used by `load_state_dict`):
        in_proj.weight              (2 * d_inner,  d_model)
        conv1d.weight               (d_inner, 1, d_conv)
        conv1d.bias                 (d_inner,)
        x_proj.weight               (dt_rank + 2*d_state,  d_inner)
        dt_proj.weight              (d_inner, dt_rank)
        dt_proj.bias                (d_inner,)
        A_log                       (d_inner, d_state)
        D                           (d_inner,)
        out_proj.weight             (d_model, d_inner)
    where d_inner = expand * d_model and dt_rank = ceil(d_model / 16) by default.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        bias: bool = False,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A_log stores log(|A|) so that A = -exp(A_log) is always negative.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        B, L, _ = x.shape
        xz = self.in_proj(x)                              # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                     # each (B, L, d_inner)

        # 1-D depth-wise causal convolution.
        x_in = x_in.transpose(1, 2)                       # (B, d_inner, L)
        x_in = self.conv1d(x_in)[..., :L]                 # drop right-padding
        x_in = F.silu(x_in).transpose(1, 2)               # (B, L, d_inner)

        # Selection projections.
        x_dbl = self.x_proj(x_in)                         # (B, L, dt_rank + 2*d_state)
        dt, B_t_all, C_t_all = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))                 # (B, L, d_inner)

        A = -torch.exp(self.A_log.float()).to(x.dtype)    # (d_inner, d_state)

        # Sequential selective scan.
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            dt_t = dt[:, t]                                # (B, d_inner)
            Bt = B_t_all[:, t]                             # (B, d_state)
            Ct = C_t_all[:, t]                             # (B, d_state)
            xi = x_in[:, t]                                # (B, d_inner)

            dA = torch.exp(dt_t.unsqueeze(-1) * A)         # (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * Bt.unsqueeze(1)      # (B, d_inner, d_state)
            h = dA * h + dB * xi.unsqueeze(-1)
            y_t = (h * Ct.unsqueeze(1)).sum(dim=-1) + self.D * xi
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                         # (B, L, d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)
