"""Joint loss = progressive classification CE  +  Sketch-RNN GMM NLL  +  pen CE.

The model produces *per-step* logits at every time step `t`.  At step `t`
the targets are:
    - class label                (constant across t for a given sketch)
    - next stroke offset (Δx, Δy) at step t+1
    - next pen state at step t+1
i.e. the recon head is a one-step predictor (causal autoregressive
teacher-forcing).  Targets at the final padded position have no successor and
are masked out.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.model import GMMParams, parse_gmm

LOG_2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------------- #
def _bivariate_gmm_log_prob(
    target: torch.Tensor, params: GMMParams
) -> torch.Tensor:
    """target: (..., 2)   params : (..., M)
    Returns per-sample log-likelihood (...,).
    """
    x = target[..., 0:1]
    y = target[..., 1:2]

    mu_x, mu_y = params.mu_x, params.mu_y
    sx, sy = params.sigma_x, params.sigma_y
    rho = params.rho

    z_x = (x - mu_x) / sx
    z_y = (y - mu_y) / sy
    one_minus_rho2 = 1.0 - rho * rho
    z = z_x * z_x - 2.0 * rho * z_x * z_y + z_y * z_y

    log_norm = -(LOG_2PI + torch.log(sx) + torch.log(sy) + 0.5 * torch.log(one_minus_rho2))
    log_comp = log_norm - z / (2.0 * one_minus_rho2)
    log_mix = params.pi + log_comp                       # (..., M)
    return torch.logsumexp(log_mix, dim=-1)              # (...,)


# --------------------------------------------------------------------------- #
def reconstruction_loss(
    out: dict,
    x: torch.Tensor,                  # (B, T, 5)
    mask: torch.Tensor,               # (B, T)  True at valid positions
    gmm_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict step t+1 from hidden at step t.

    Returns (gmm_nll, pen_ce, total_recon).
    """
    gmm_params = parse_gmm(out["gmm"], gmm_components)        # per-step
    pen_logits = out["pen_logits"]

    # Shift: target_{t} = input_{t+1}; valid where both t and t+1 are valid.
    target_xy = x[:, 1:, :2]                                  # (B, T-1, 2)
    target_pen = x[:, 1:, 2:5].argmax(dim=-1)                 # (B, T-1)
    pred_xy_params = GMMParams(
        gmm_params.pi[:, :-1],
        gmm_params.mu_x[:, :-1],
        gmm_params.mu_y[:, :-1],
        gmm_params.sigma_x[:, :-1],
        gmm_params.sigma_y[:, :-1],
        gmm_params.rho[:, :-1],
    )
    pred_pen = pen_logits[:, :-1]                             # (B, T-1, 3)
    valid = mask[:, 1:]                                       # both step t and t+1 valid

    log_p = _bivariate_gmm_log_prob(target_xy, pred_xy_params)  # (B, T-1)
    gmm_nll = -(log_p * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

    pen_ce_full = F.cross_entropy(
        pred_pen.reshape(-1, 3), target_pen.reshape(-1), reduction="none"
    ).reshape_as(target_pen)
    pen_ce = (pen_ce_full * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

    return gmm_nll, pen_ce, gmm_nll + pen_ce


# --------------------------------------------------------------------------- #
def progressive_cls_loss(
    cls_logits: torch.Tensor,         # (B, T, C)
    label: torch.Tensor,              # (B,)
    mask: torch.Tensor,               # (B, T)
    scheme: str = "per_step",
    k_cuts: int = 4,
    step_weighting: str = "uniform",
) -> torch.Tensor:
    B, T, C = cls_logits.shape
    target = label[:, None].expand(B, T)
    ce = F.cross_entropy(
        cls_logits.reshape(-1, C), target.reshape(-1), reduction="none"
    ).reshape(B, T)                    # (B, T)

    if scheme == "per_step":
        valid = mask.float()
        if step_weighting == "linear":
            w = torch.arange(1, T + 1, device=cls_logits.device, dtype=valid.dtype)
            w = w[None, :].expand(B, T) * valid
        else:
            w = valid
        w_sum = w.sum().clamp(min=1.0)
        return (ce * w).sum() / w_sum

    if scheme == "last_step":
        last_idx = mask.long().sum(dim=1).clamp(min=1) - 1     # (B,)
        gather = ce.gather(1, last_idx[:, None]).squeeze(1)    # (B,)
        return gather.mean()

    if scheme == "k_cut":
        if k_cuts <= 0:
            raise ValueError("k_cuts must be > 0")
        # Sample k random valid prefix end-positions per sample.
        lengths = mask.long().sum(dim=1).clamp(min=1)          # (B,)
        # Uniform in [0, length-1].
        rand = torch.rand(B, k_cuts, device=cls_logits.device)
        cuts = (rand * lengths[:, None].float()).long().clamp(max=T - 1)
        ce_cuts = ce.gather(1, cuts)                           # (B, k)
        return ce_cuts.mean()

    raise ValueError(f"Unknown progressive scheme: {scheme}")


# --------------------------------------------------------------------------- #
def joint_loss(
    out: dict,
    batch: dict,
    *,
    gmm_components: int,
    lambda_cls: float = 1.0,
    lambda_rec: float = 1.0,
    progressive: str = "per_step",
    k_cuts: int = 4,
    step_weighting: str = "uniform",
) -> dict:
    x = batch["x"]
    mask = batch["mask"]
    label = batch["label"]

    if lambda_cls > 0.0:
        l_cls = progressive_cls_loss(
            out["cls_logits"],
            label,
            mask,
            scheme=progressive,
            k_cuts=k_cuts,
            step_weighting=step_weighting,
        )
    else:
        l_cls = torch.zeros((), device=x.device)

    if lambda_rec > 0.0:
        gmm_nll, pen_ce, l_rec = reconstruction_loss(out, x, mask, gmm_components)
    else:
        gmm_nll = pen_ce = l_rec = torch.zeros((), device=x.device)

    total = lambda_cls * l_cls + lambda_rec * l_rec
    return {
        "loss": total,
        "loss_cls": l_cls.detach(),
        "loss_gmm": gmm_nll.detach(),
        "loss_pen": pen_ce.detach(),
        "loss_rec": l_rec.detach(),
    }
