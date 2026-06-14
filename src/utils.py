"""Shared utilities: seeding, EMA, deep-merge, scheduler, plotting helpers."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; values in `override` win."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


# --------------------------------------------------------------------------- #
class CosineWarmup:
    """Cosine schedule with linear warmup, applied per-step."""

    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup: int, total: int, min_lr: float = 1e-6
    ) -> None:
        self.optimizer = optimizer
        self.warmup = max(1, int(warmup))
        self.total = max(self.warmup + 1, int(total))
        self.min_lr = float(min_lr)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        s = self.step_num
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            if s <= self.warmup:
                lr = base * s / self.warmup
            else:
                t = (s - self.warmup) / max(1, self.total - self.warmup)
                t = min(1.0, t)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1.0 + math.cos(math.pi * t))
            g["lr"] = lr


# --------------------------------------------------------------------------- #
class EMA:
    """Exponential moving average of model parameters (kept on the same device)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            sv = self.shadow[k]
            if v.dtype.is_floating_point:
                sv.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                sv.copy_(v)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict) -> None:
        for k, v in sd.items():
            self.shadow[k] = v.clone()

    def copy_to(self, model: torch.nn.Module) -> dict:
        """Swap EMA weights into model; returns the previous state for restoration."""
        prev = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return prev


# --------------------------------------------------------------------------- #
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig, base_path: str | Path) -> None:
    """Save a matplotlib figure as both PNG and PDF (research convention)."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=200)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


# --------------------------------------------------------------------------- #
def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):  # noqa: ANN001
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


# --------------------------------------------------------------------------- #
def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_summary() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "git_sha": git_sha(),
        "pid": os.getpid(),
    }
    return info
