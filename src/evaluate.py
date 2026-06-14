"""Full research-grade evaluation of a SketchMamba checkpoint.

Computes:
  - Top-1 / Top-5 final accuracy, per-class F1, confusion matrix.
  - Progressive accuracy curve over prefix ratios + scalar AUC + earliness@0.9.
  - GMM NLL, pen-state accuracy + macro-F1.
  - Stroke continuation: ADE / FDE (greedy + sampled) and Chamfer at 128².
  - Efficiency: params, FLOPs (DeepSpeed profiler), peak VRAM, GPU latency.

Saves figures to results/figures/<run> as both PNG and PDF and metrics
JSONs under results/metrics/<run>/.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import PackedSketchDataset, collate_pad, stratified_subset_indices
from src.losses import _bivariate_gmm_log_prob
from src.model import build_model, count_parameters, parse_gmm
from src.utils import device_auto, ensure_dir, save_figure, save_json, set_seed

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# --------------------------------------------------------------------------- #
def load_checkpoint(ckpt_path: Path, device: torch.device, prefer_ema: bool = True):
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = state["config"]
    model_cfg = dict(cfg["model"])
    model_cfg.setdefault("max_len", cfg["data"]["max_len"] + 8)
    model = build_model(model_cfg).to(device)

    sd = state["ema"] if prefer_ema and "ema" in state else state["model"]
    # Strip torch.compile prefix if present.
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg, state


# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_per_step_predictions(model, loader, device, num_classes: int):
    """Returns:
        all_logits_at_ratio: (N, R, C) logits at chosen ratios
        all_labels         : (N,)
        ratios             : list[float]
    Plus auxiliary tensors for recon metrics.
    """
    pass  # implemented inline in metric routines below


# --------------------------------------------------------------------------- #
def progressive_accuracy(model, loader, device, ratios, num_classes):
    """For each sample, read the per-step class logits at floor(r*L)-1 and
    accumulate Top-1 accuracy."""
    R = len(ratios)
    correct = np.zeros(R, dtype=np.int64)
    total = np.zeros(R, dtype=np.int64)
    top5_correct = np.zeros(R, dtype=np.int64)

    for batch in tqdm(loader, desc="progressive cls"):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch["x"], mask=batch["mask"])
        cls_logits = out["cls_logits"].float()                  # (B, T, C)
        lengths = batch["mask"].long().sum(dim=1)               # (B,)
        labels = batch["label"]                                  # (B,)
        for r_idx, r in enumerate(ratios):
            t_idx = (lengths.float() * r).long().clamp(min=1) - 1
            logits = cls_logits.gather(
                1, t_idx.view(-1, 1, 1).expand(-1, 1, num_classes)
            ).squeeze(1)
            pred = logits.argmax(dim=-1)
            correct[r_idx] += int((pred == labels).sum())
            top5 = logits.topk(5, dim=-1).indices
            top5_correct[r_idx] += int((top5 == labels[:, None]).any(dim=-1).sum())
            total[r_idx] += int(labels.size(0))
    acc = correct / np.maximum(total, 1)
    top5 = top5_correct / np.maximum(total, 1)
    return acc, top5


def auc_progressive(ratios, acc):
    """Trapezoidal AUC over r in [r_min, r_max]; normalised to [0, 1]."""
    r = np.asarray(ratios, dtype=np.float64)
    a = np.asarray(acc, dtype=np.float64)
    return float(np.trapz(a, r) / (r[-1] - r[0]))


def earliness_at_threshold(ratios, acc, frac=0.9):
    """Smallest r such that acc(r) >= frac * acc(r=1)."""
    target = frac * acc[-1]
    for r, a in zip(ratios, acc):
        if a >= target:
            return float(r)
    return float(ratios[-1])


# --------------------------------------------------------------------------- #
@torch.no_grad()
def final_metrics(model, loader, device, num_classes: int, gmm_components: int):
    """Final-step Top-1/5, per-class F1, confusion matrix; also pen-state metrics
    and GMM NLL aggregated over the whole loader."""
    all_pred, all_label = [], []
    pen_pred, pen_target = [], []
    gmm_nll_sum = 0.0
    gmm_nll_n = 0

    for batch in tqdm(loader, desc="final metrics"):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch["x"], mask=batch["mask"])
        cls_logits = out["cls_logits"].float()
        lengths = batch["mask"].long().sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        last_logits = cls_logits.gather(
            1, last_idx.view(-1, 1, 1).expand(-1, 1, num_classes)
        ).squeeze(1)
        all_pred.append(last_logits.argmax(dim=-1).cpu().numpy())
        all_label.append(batch["label"].cpu().numpy())

        # Pen + GMM at every valid (t, t+1) position.
        x = batch["x"]
        mask = batch["mask"]
        valid = mask[:, 1:]
        gmm = parse_gmm(out["gmm"].float(), gmm_components)
        gmm = type(gmm)(
            gmm.pi[:, :-1], gmm.mu_x[:, :-1], gmm.mu_y[:, :-1],
            gmm.sigma_x[:, :-1], gmm.sigma_y[:, :-1], gmm.rho[:, :-1],
        )
        target_xy = x[:, 1:, :2]
        log_p = _bivariate_gmm_log_prob(target_xy, gmm)         # (B, T-1)
        gmm_nll_sum += float(-(log_p * valid.float()).sum())
        gmm_nll_n += int(valid.sum())

        pen_logits = out["pen_logits"][:, :-1].float()
        pen_t = x[:, 1:, 2:5].argmax(dim=-1)
        pp = pen_logits.argmax(dim=-1)
        m = valid
        pen_pred.append(pp[m].cpu().numpy())
        pen_target.append(pen_t[m].cpu().numpy())

    pred = np.concatenate(all_pred)
    label = np.concatenate(all_label)
    top1 = float((pred == label).mean())
    f1_per_class = f1_score(label, pred, average=None, labels=np.arange(num_classes), zero_division=0)
    f1_macro = float(f1_score(label, pred, average="macro", labels=np.arange(num_classes), zero_division=0))
    cm = confusion_matrix(label, pred, labels=np.arange(num_classes))

    pen_pred = np.concatenate(pen_pred)
    pen_target = np.concatenate(pen_target)
    pen_acc = float((pen_pred == pen_target).mean())
    pen_f1_macro = float(f1_score(pen_target, pen_pred, average="macro", labels=[0, 1, 2], zero_division=0))

    return {
        "top1": top1,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
        "pen_acc": pen_acc,
        "pen_f1_macro": pen_f1_macro,
        "gmm_nll_per_step": gmm_nll_sum / max(1, gmm_nll_n),
    }


# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_continuation(
    model,
    prefix: torch.Tensor,           # (1, p, 5)
    n_steps: int,
    gmm_components: int,
    device: torch.device,
    temperature: float = 0.4,
    greedy: bool = False,
) -> torch.Tensor:
    """Autoregressively sample `n_steps` future tokens from the model.

    Returns a tensor of shape (n_steps, 5) — the appended continuation only.
    """
    model.eval()
    seq = prefix.clone().to(device)
    out_steps = []
    for _ in range(n_steps):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(seq)
        gmm = parse_gmm(out["gmm"].float()[:, -1:], gmm_components)   # last step
        pen_logits = out["pen_logits"].float()[:, -1]                 # (1, 3)

        if greedy:
            comp = gmm.pi.argmax(dim=-1)                              # (1, 1)
            i = comp[..., 0]                                          # (1,)
            mu_x = gmm.mu_x[..., 0, :].gather(-1, i[:, None]).squeeze(-1)
            mu_y = gmm.mu_y[..., 0, :].gather(-1, i[:, None]).squeeze(-1)
            dx, dy = mu_x, mu_y
            pen = pen_logits.argmax(dim=-1)
        else:
            # Temperature on mixture weights.
            logits = gmm.pi[:, 0] / max(temperature, 1e-3)
            comp = torch.distributions.Categorical(logits=logits).sample()    # (1,)
            sel = comp[:, None, None]
            mu_x = gmm.mu_x[:, 0].gather(-1, comp[:, None]).squeeze(-1)
            mu_y = gmm.mu_y[:, 0].gather(-1, comp[:, None]).squeeze(-1)
            sx = gmm.sigma_x[:, 0].gather(-1, comp[:, None]).squeeze(-1) * math.sqrt(temperature)
            sy = gmm.sigma_y[:, 0].gather(-1, comp[:, None]).squeeze(-1) * math.sqrt(temperature)
            rho = gmm.rho[:, 0].gather(-1, comp[:, None]).squeeze(-1)
            eps1 = torch.randn_like(mu_x)
            eps2 = torch.randn_like(mu_y)
            dx = mu_x + sx * eps1
            dy = mu_y + sy * (rho * eps1 + torch.sqrt(1 - rho * rho) * eps2)
            pen_probs = F.softmax(pen_logits / max(temperature, 1e-3), dim=-1)
            pen = torch.distributions.Categorical(probs=pen_probs).sample()

        token = torch.zeros(1, 1, 5, device=device, dtype=seq.dtype)
        token[..., 0] = dx
        token[..., 1] = dy
        token[0, 0, 2 + int(pen)] = 1.0
        out_steps.append(token.squeeze(0).squeeze(0).clone())
        seq = torch.cat([seq, token], dim=1)
        if int(pen) == 2:  # end-of-sketch
            break
    return torch.stack(out_steps, dim=0) if out_steps else torch.zeros(0, 5, device=device)


# --------------------------------------------------------------------------- #
def _stroke5_to_polyline(
    stroke5: np.ndarray,
    dx_std: float,
    dy_std: float,
    start_xy: tuple[float, float] = (0.0, 0.0),
) -> list[np.ndarray]:
    """Convert stroke-5 to absolute (x, y) polylines (one per pen-down stroke)."""
    pts = []
    cur = []
    x, y = float(start_xy[0]), float(start_xy[1])
    for row in stroke5:
        dx = row[0] * dx_std
        dy = row[1] * dy_std
        x += float(dx)
        y += float(dy)
        cur.append((x, y))
        # 0=p1 down, 1=p2 up, 2=p3 end
        state = int(np.argmax(row[2:5]))
        if state != 0:
            if len(cur) > 1:
                pts.append(np.asarray(cur, dtype=np.float32))
            cur = [(x, y)]
            if state == 2:
                break
    if len(cur) > 1:
        pts.append(np.asarray(cur, dtype=np.float32))
    return pts


def _rasterize_polylines(polylines: list[np.ndarray], size: int = 128) -> np.ndarray:
    if not polylines:
        return np.zeros((size, size), dtype=np.uint8)
    all_pts = np.concatenate(polylines, axis=0)
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    span = max(float((hi - lo).max()), 1.0)
    img = np.zeros((size, size), dtype=np.uint8)
    margin = 4
    scale = (size - 2 * margin) / span
    for pl in polylines:
        p = (pl - lo) * scale + margin
        for i in range(len(p) - 1):
            _draw_line(img, p[i], p[i + 1])
    return img


def _draw_line(img: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> None:
    """Bresenham-ish line drawing."""
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    H, W = img.shape
    while True:
        if 0 <= x0 < W and 0 <= y0 < H:
            img[y0, x0] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Chamfer between two binary images (set of foreground pixels)."""
    pa = np.argwhere(a > 0).astype(np.float32)
    pb = np.argwhere(b > 0).astype(np.float32)
    if pa.size == 0 or pb.size == 0:
        return float("nan")
    # KD-tree-free brute force; subsample for speed.
    if pa.shape[0] > 1024:
        pa = pa[np.random.choice(pa.shape[0], 1024, replace=False)]
    if pb.shape[0] > 1024:
        pb = pb[np.random.choice(pb.shape[0], 1024, replace=False)]
    d_ab = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)).min(axis=1).mean()
    d_ba = np.sqrt(((pb[:, None, :] - pa[None, :, :]) ** 2).sum(-1)).min(axis=1).mean()
    return float((d_ab + d_ba) / 2.0)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def completion_metrics(
    model,
    dataset: PackedSketchDataset,
    indices: np.ndarray,
    device,
    gmm_components: int,
    prefix_ratios=(0.25, 0.5, 0.75),
    temperatures=(0.1, 0.4, 0.8),
    n_samples: int = 20,
    chamfer_resolution: int = 128,
):
    dx_std, dy_std = dataset.dx_std, dataset.dy_std
    out_metrics = {
        "ade": {f"r={r}": {} for r in prefix_ratios},
        "fde": {f"r={r}": {} for r in prefix_ratios},
        "chamfer": {f"r={r}": {} for r in prefix_ratios},
    }
    # store raw samples for aggregation
    accum = {(r, t): {"ade": [], "fde": [], "chamfer": []} for r in prefix_ratios for t in (("greedy",) + tuple(temperatures))}

    for i in tqdm(indices, desc="completions"):
        item = dataset[int(i)]
        x = item["x"].numpy()                  # (T, 5)
        T = x.shape[0]
        gt_polylines = _stroke5_to_polyline(x, dx_std, dy_std)
        gt_img = _rasterize_polylines(gt_polylines, size=chamfer_resolution)

        for r in prefix_ratios:
            p = max(2, int(round(r * T)))
            prefix = torch.from_numpy(x[:p]).unsqueeze(0)
            n_remaining = T - p

            # Greedy
            cont = sample_continuation(
                model, prefix, n_steps=n_remaining,
                gmm_components=gmm_components, device=device, greedy=True,
            ).cpu().numpy()
            full = np.concatenate([x[:p], cont], axis=0)
            ade, fde = _ade_fde(x[p:p + len(cont)], cont, dx_std, dy_std)
            cham = _chamfer_distance(
                gt_img, _rasterize_polylines(_stroke5_to_polyline(full, dx_std, dy_std), chamfer_resolution)
            )
            accum[(r, "greedy")]["ade"].append(ade)
            accum[(r, "greedy")]["fde"].append(fde)
            accum[(r, "greedy")]["chamfer"].append(cham)

            for tau in temperatures:
                ades, fdes, chams = [], [], []
                for _ in range(n_samples):
                    cont = sample_continuation(
                        model, prefix, n_steps=n_remaining,
                        gmm_components=gmm_components, device=device,
                        temperature=tau, greedy=False,
                    ).cpu().numpy()
                    if len(cont) == 0:
                        continue
                    full = np.concatenate([x[:p], cont], axis=0)
                    ade, fde = _ade_fde(x[p:p + len(cont)], cont, dx_std, dy_std)
                    cham = _chamfer_distance(
                        gt_img,
                        _rasterize_polylines(_stroke5_to_polyline(full, dx_std, dy_std), chamfer_resolution),
                    )
                    ades.append(ade)
                    fdes.append(fde)
                    chams.append(cham)
                if ades:
                    accum[(r, tau)]["ade"].append(float(np.mean(ades)))
                    accum[(r, tau)]["fde"].append(float(np.mean(fdes)))
                    accum[(r, tau)]["chamfer"].append(float(np.nanmean(chams)))

    # Aggregate mean ± 95% CI (1.96 * sem).
    summary = {}
    for (r, t), vals in accum.items():
        key = f"r={r}_t={t}"
        summary[key] = {}
        for m, arr in vals.items():
            arr = np.asarray([v for v in arr if not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
            if arr.size == 0:
                summary[key][m] = {"mean": float("nan"), "ci95": float("nan"), "n": 0}
            else:
                mean = float(arr.mean())
                sem = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
                summary[key][m] = {"mean": mean, "ci95": 1.96 * sem, "n": int(arr.size)}
    return summary


def _ade_fde(gt: np.ndarray, pred: np.ndarray, dx_std: float, dy_std: float):
    """Cumulative Δx,Δy → absolute coordinates, then ADE/FDE in canvas pixels."""
    n = min(len(gt), len(pred))
    if n == 0:
        return float("nan"), float("nan")
    g = gt[:n, :2].astype(np.float64) * np.array([dx_std, dy_std])
    p = pred[:n, :2].astype(np.float64) * np.array([dx_std, dy_std])
    g_abs = np.cumsum(g, axis=0)
    p_abs = np.cumsum(p, axis=0)
    d = np.sqrt(((g_abs - p_abs) ** 2).sum(-1))
    return float(d.mean()), float(d[-1])


# --------------------------------------------------------------------------- #
def efficiency(model, device, max_len: int) -> dict:
    info = {"params": int(count_parameters(model)), "device": str(device)}
    # FLOPs via DeepSpeed if available
    try:
        from deepspeed.profiling.flops_profiler import get_model_profile

        x = torch.zeros(1, max_len, 5, device=device)
        flops, macs, _ = get_model_profile(
            model, kwargs={"x": x}, print_profile=False, as_string=False, warm_up=1,
        )
        info["flops"] = int(flops)
        info["macs"] = int(macs)
    except Exception as e:  # noqa: BLE001
        info["flops_error"] = repr(e)

    # Latency at multiple lengths
    latencies = {}
    for T in (50, 100, 150, max_len):
        x = torch.zeros(1, T, 5, device=device)
        # warm-up
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        N = 50
        for _ in range(N):
            with torch.no_grad():
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies[f"T={T}"] = {"ms_per_seq": 1000.0 * (time.time() - t0) / N}
    info["latency"] = latencies

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        x = torch.zeros(1, max_len, 5, device=device)
        with torch.no_grad():
            _ = model(x)
        info["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    return info


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_progressive(ratios, acc, top5, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ratios, acc, marker="o", linewidth=2, label="Top-1")
    ax.plot(ratios, top5, marker="s", linewidth=2, label="Top-5")
    ax.set_xlabel("Stroke completion ratio  $r$")
    ax.set_ylabel("Accuracy")
    ax.set_title("Progressive classification accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, fig_dir / "progressive_accuracy")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: list[str], fig_dir: Path) -> None:
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(
        cm_norm, ax=ax, cmap="viridis", square=True,
        xticklabels=classes, yticklabels=classes, cbar_kws={"shrink": 0.7},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised confusion matrix (final-step)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    save_figure(fig, fig_dir / "confusion_matrix")
    plt.close(fig)


def plot_loss_curves(csv_path: Path, fig_dir: Path) -> None:
    if not csv_path.exists():
        return
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["epoch"], df["train_loss"], label="train", linewidth=2)
    ax.plot(df["epoch"], df["val_loss"], label="val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Joint loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, fig_dir / "loss_curves")
    plt.close(fig)


def plot_latency(latency: dict, fig_dir: Path) -> None:
    Ts, ms = [], []
    for k, v in latency.items():
        Ts.append(int(k.split("=")[1]))
        ms.append(float(v["ms_per_seq"]))
    order = np.argsort(Ts)
    Ts = np.array(Ts)[order]
    ms = np.array(ms)[order]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(Ts, ms, marker="o", linewidth=2)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Latency (ms / sequence, GPU)")
    ax.set_title("Forward-pass latency")
    ax.grid(True, alpha=0.3)
    save_figure(fig, fig_dir / "latency_vs_length")
    plt.close(fig)


def plot_qualitative(model, dataset, device, gmm_components, classes, fig_dir, per_class=1, n_samples=3):
    rng = np.random.default_rng(0)
    labels = dataset.index["label"].astype(np.int64)
    chosen = []
    for c in range(len(classes)):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        chosen.extend(rng.choice(idx, size=min(per_class, idx.size), replace=False).tolist())
    rows = len(chosen)
    if rows == 0:
        return
    cols = 1 + n_samples
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    if rows == 1:
        axes = axes[None, :]
    for r, i in enumerate(chosen):
        item = dataset[int(i)]
        x = item["x"].numpy()
        T = x.shape[0]
        p = max(2, T // 2)
        prefix = torch.from_numpy(x[:p]).unsqueeze(0)
        ax = axes[r, 0]
        _draw_polylines(ax, _stroke5_to_polyline(x, dataset.dx_std, dataset.dy_std), color="black")
        _draw_polylines(ax, _stroke5_to_polyline(x[:p], dataset.dx_std, dataset.dy_std), color="C0", lw=2)
        ax.set_title(f"{classes[int(item['label'])]}  (prefix)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        # Compute the end-of-prefix absolute position so continuations attach correctly.
        prefix_dx = (x[1:p, 0] * dataset.dx_std).sum()
        prefix_dy = (x[1:p, 1] * dataset.dy_std).sum()
        for s in range(n_samples):
            cont = sample_continuation(
                model, prefix, n_steps=T - p, gmm_components=gmm_components,
                device=device, temperature=0.4, greedy=False,
            ).cpu().numpy()
            ax2 = axes[r, 1 + s]
            _draw_polylines(ax2, _stroke5_to_polyline(x[:p], dataset.dx_std, dataset.dy_std), color="C0", lw=2)
            _draw_polylines(
                ax2,
                _stroke5_to_polyline(
                    cont, dataset.dx_std, dataset.dy_std,
                    start_xy=(float(prefix_dx), float(prefix_dy)),
                ),
                color=f"C{s+2}", lw=1.5,
            )
            ax2.set_title(f"sample {s+1}", fontsize=8)
            ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_aspect("equal")
    save_figure(fig, fig_dir / "qualitative_completions")
    plt.close(fig)


def _draw_polylines(ax, polylines, color="black", lw=1.0):
    for pl in polylines:
        ax.plot(pl[:, 0], -pl[:, 1], color=color, linewidth=lw)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--completion_n", type=int, default=500,
                    help="Number of test sketches used for completion metrics.")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = device_auto()
    model, ckpt_cfg, _state = load_checkpoint(args.ckpt, device, prefer_ema=True)
    cfg = ckpt_cfg                         # use the exact config the checkpoint was trained with
    run_name = cfg["run_name"]

    metrics_dir = ensure_dir((args.out_dir or Path(cfg["eval"]["metrics_dir"])) / run_name)
    figures_dir = ensure_dir(Path(cfg["eval"]["figures_dir"]) / run_name)

    ds = PackedSketchDataset(
        packed_dir=cfg["data"]["packed_dir"],
        split=args.split,
        max_len=cfg["data"]["max_len"],
        augment=False,
    )
    loader = DataLoader(
        ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate_pad,
        pin_memory=True, persistent_workers=cfg["data"]["num_workers"] > 0,
    )

    classes = ds.meta.classes
    num_classes = ds.meta.num_classes

    # 1) Progressive accuracy
    ratios = cfg["eval"]["prefix_ratios"]
    acc, top5 = progressive_accuracy(model, loader, device, ratios, num_classes)
    auc = auc_progressive(ratios, acc)
    earl = earliness_at_threshold(ratios, acc, frac=0.9)
    plot_progressive(ratios, acc, top5, figures_dir)

    # 2) Final + per-class + pen + GMM NLL
    fm = final_metrics(model, loader, device, num_classes, cfg["model"]["gmm_components"])
    plot_confusion(np.asarray(fm["confusion_matrix"]), classes, figures_dir)

    # 3) Completion metrics on a stratified subset
    labels = ds.index["label"].astype(np.int64)
    subset = stratified_subset_indices(labels, n_total=args.completion_n, seed=cfg["seed"])
    completion = completion_metrics(
        model, ds, subset, device, cfg["model"]["gmm_components"],
        prefix_ratios=tuple(cfg["eval"]["completion_prefix_ratios"]),
        temperatures=tuple(cfg["eval"]["completion_temperatures"]),
        n_samples=cfg["eval"]["completion_samples"],
        chamfer_resolution=cfg["eval"]["chamfer_resolution"],
    )

    # 4) Efficiency
    eff = efficiency(model, device, cfg["data"]["max_len"])
    plot_latency(eff["latency"], figures_dir)

    # 5) Loss curves
    plot_loss_curves(Path(cfg["train"]["log_dir"]) / run_name / "train.csv", figures_dir)

    # 6) Qualitative
    plot_qualitative(
        model, ds, device, cfg["model"]["gmm_components"], classes, figures_dir,
        per_class=cfg["eval"]["qualitative_per_class"], n_samples=3,
    )

    summary = {
        "run_name": run_name,
        "split": args.split,
        "classes": classes,
        "ratios": ratios,
        "progressive_top1": acc.tolist(),
        "progressive_top5": top5.tolist(),
        "auc_progressive": auc,
        "earliness_at_0p9": earl,
        "final": {k: v for k, v in fm.items() if k not in ("confusion_matrix", "f1_per_class")},
        "f1_per_class": fm["f1_per_class"],
        "completion": completion,
        "efficiency": eff,
    }
    save_json(metrics_dir / "metrics.json", summary)
    # Confusion matrix saved separately (large).
    save_json(metrics_dir / "confusion_matrix.json", {"classes": classes, "matrix": fm["confusion_matrix"]})

    print("\n=== Summary ===")
    print(f"  Top-1 (final)        : {fm['top1']:.4f}")
    print(f"  Top-1 AUC            : {auc:.4f}")
    print(f"  Earliness@0.9        : {earl:.2f}")
    print(f"  Pen accuracy         : {fm['pen_acc']:.4f}")
    print(f"  GMM NLL / step       : {fm['gmm_nll_per_step']:.4f}")
    print(f"  Params               : {eff['params']:,}")
    print(f"  Wrote                : {metrics_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
