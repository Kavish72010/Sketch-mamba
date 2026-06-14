"""Main training entry point for SketchMamba.

Usage:
    python -m src.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data import (
    PackedSketchDataset,
    collate_pad,
    make_loader,
    stratified_subset_indices,
)
from src.losses import joint_loss
from src.model import build_model, count_parameters
from src.utils import (
    CosineWarmup,
    EMA,
    config_hash,
    device_auto,
    ensure_dir,
    env_summary,
    save_json,
    set_seed,
)


# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
def build_val_loader(cfg: dict) -> DataLoader:
    ds = PackedSketchDataset(
        packed_dir=cfg["data"]["packed_dir"],
        split="valid",
        max_len=cfg["data"]["max_len"],
        augment=False,
    )
    labels = ds.index["label"].astype(np.int64)
    n = int(cfg["train"]["val_subset_size"])
    idx = stratified_subset_indices(labels, n_total=n, seed=cfg["seed"])
    sub = Subset(ds, idx.tolist())
    return DataLoader(
        sub,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_pad,
        pin_memory=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )


# --------------------------------------------------------------------------- #
def evaluate_subset(model, loader, cfg, device) -> dict:
    model.eval()
    totals = {"loss": 0.0, "loss_cls": 0.0, "loss_gmm": 0.0, "loss_pen": 0.0}
    n_batches = 0
    cls_correct_full = 0
    cls_total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(batch["x"], mask=batch["mask"])
                losses = joint_loss(
                    out,
                    batch,
                    gmm_components=cfg["model"]["gmm_components"],
                    lambda_cls=cfg["loss"]["lambda_cls"],
                    lambda_rec=cfg["loss"]["lambda_rec"],
                    progressive=cfg["loss"]["progressive"],
                    k_cuts=cfg["loss"]["k_cuts"],
                    step_weighting=cfg["loss"]["step_weighting"],
                )
            for k in totals:
                totals[k] += float(losses[k])
            n_batches += 1
            # Final-step Top-1 accuracy
            last = batch["mask"].long().sum(dim=1).clamp(min=1) - 1
            logits_last = out["cls_logits"].gather(
                1, last.view(-1, 1, 1).expand(-1, 1, out["cls_logits"].size(-1))
            ).squeeze(1)
            pred = logits_last.argmax(dim=-1)
            cls_correct_full += int((pred == batch["label"]).sum())
            cls_total += int(batch["label"].size(0))
    avg = {k: v / max(1, n_batches) for k, v in totals.items()}
    avg["acc_top1_final"] = cls_correct_full / max(1, cls_total)
    return avg


# --------------------------------------------------------------------------- #
def train(cfg: dict) -> Path:
    set_seed(cfg["seed"])
    device = device_auto()
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA device required (mamba-ssm has no CPU implementation)."
        )

    run_name = cfg["run_name"]
    ckpt_dir = ensure_dir(Path(cfg["train"]["ckpt_dir"]) / run_name)
    log_dir = ensure_dir(Path(cfg["train"]["log_dir"]) / run_name)

    # ---- Logging meta ----
    meta = {
        "config": cfg,
        "config_hash": config_hash(cfg),
        "env": env_summary(),
    }
    save_json(ckpt_dir / "run_meta.json", meta)

    # ---- Data ----
    train_loader = make_loader(
        packed_dir=cfg["data"]["packed_dir"],
        split="train",
        batch_size=cfg["data"]["batch_size"],
        max_len=cfg["data"]["max_len"],
        num_workers=cfg["data"]["num_workers"],
        bucket_size_multiplier=cfg["data"]["bucket_size_multiplier"],
        augment=cfg["data"]["augment"],
        shuffle=True,
        drop_last=True,
        seed=cfg["seed"],
        aug_scale_range=tuple(cfg["data"]["aug_scale_range"]),
        aug_stroke_dropout=cfg["data"]["aug_stroke_dropout"],
    )
    val_loader = build_val_loader(cfg)

    # ---- Model ----
    model_cfg = dict(cfg["model"])
    model_cfg.setdefault("max_len", cfg["data"]["max_len"] + 8)
    model = build_model(model_cfg).to(device)
    print(f"Model: {model_cfg['type']}    params = {count_parameters(model):,}")

    if cfg["train"].get("compile", False):
        model = torch.compile(model, mode="default", dynamic=True)

    # ---- Optimiser ----
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or n.endswith(".bias") else decay).append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["train"]["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["train"]["lr"],
        betas=(cfg["train"]["beta1"], cfg["train"]["beta2"]),
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["train"]["max_epochs"]
    sched = CosineWarmup(optim, cfg["train"]["warmup_steps"], total_steps)

    ema = EMA(model, decay=cfg["train"]["ema_decay"])

    # ---- Optional resume from sketchmamba_last.pt ----
    last_ckpt = ckpt_dir / "sketchmamba_last.pt"
    resume_epoch = 0
    best_val = float("inf")
    bad_epochs = 0
    global_step = 0
    if last_ckpt.exists():
        print(f"Found existing checkpoint at {last_ckpt}; attempting resume.")
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        saved_hash = state.get("config_hash")
        if saved_hash is not None and saved_hash != config_hash(cfg):
            raise SystemExit(
                f"Refusing to resume: config_hash changed.\n"
                f"  saved : {saved_hash}\n  current: {config_hash(cfg)}\n"
                f"Delete {last_ckpt} (and ideally the whole {ckpt_dir}) "
                f"if you want to start a fresh run with the new config."
            )
        # Align `_orig_mod.` prefix to whatever the current model expects
        # (it differs depending on whether torch.compile is on this run).
        cur_keys = list(model.state_dict().keys())
        cur_prefixed = any(k.startswith("_orig_mod.") for k in cur_keys)
        def _align(sd):
            saved_prefixed = any(k.startswith("_orig_mod.") for k in sd)
            if cur_prefixed == saved_prefixed:
                return sd
            if cur_prefixed:
                return {"_orig_mod." + k: v for k, v in sd.items()}
            return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(_align(state["model"]), strict=True)
        # Rebuild EMA shadow from scratch so no stale keys linger.
        ema.shadow = {k: v.clone() for k, v in _align(state["ema"]).items()}
        if "optim" in state:
            optim.load_state_dict(state["optim"])
        resume_epoch = int(state.get("epoch", 0))
        # Old checkpoints lack global_step; approximate from epoch.
        global_step = int(state.get("global_step", resume_epoch * steps_per_epoch))
        sched.step_num = global_step
        best_val = float(state.get("best_val", float("inf")))
        bad_epochs = int(state.get("bad_epochs", 0))
        print(
            f"Resuming from epoch {resume_epoch + 1}  "
            f"(global_step={global_step}, best_val={best_val:.4f}, "
            f"bad_epochs={bad_epochs})."
        )

    # ---- CSV log (append on resume, write header only on a fresh run) ----
    csv_path = log_dir / "train.csv"
    fresh_csv = resume_epoch == 0 or not csv_path.exists()
    csv_file = open(csv_path, "w" if fresh_csv else "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if fresh_csv:
        writer.writerow(
            [
                "epoch", "step", "lr", "train_loss", "train_loss_cls",
                "train_loss_gmm", "train_loss_pen", "val_loss", "val_loss_cls",
                "val_loss_gmm", "val_loss_pen", "val_acc_top1_final",
                "wall_time_s",
            ]
        )

    t_start = time.time()

    for epoch in range(resume_epoch + 1, cfg["train"]["max_epochs"] + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        model.train()
        running = {"loss": 0.0, "loss_cls": 0.0, "loss_gmm": 0.0, "loss_pen": 0.0}
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg['train']['max_epochs']}")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch["x"], mask=batch["mask"])
                losses = joint_loss(
                    out,
                    batch,
                    gmm_components=cfg["model"]["gmm_components"],
                    lambda_cls=cfg["loss"]["lambda_cls"],
                    lambda_rec=cfg["loss"]["lambda_rec"],
                    progressive=cfg["loss"]["progressive"],
                    k_cuts=cfg["loss"]["k_cuts"],
                    step_weighting=cfg["loss"]["step_weighting"],
                )
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optim.step()
            sched.step()
            ema.update(model)

            global_step += 1
            for k in running:
                running[k] += float(losses[k])
            n += 1
            if global_step % 50 == 0:
                pbar.set_postfix(
                    loss=f"{running['loss']/n:.3f}",
                    cls=f"{running['loss_cls']/n:.3f}",
                    rec=f"{(running['loss_gmm']+running['loss_pen'])/n:.3f}",
                    lr=f"{optim.param_groups[0]['lr']:.2e}",
                )

        # Per-epoch validation under EMA weights.
        prev = ema.copy_to(model)
        val = evaluate_subset(model, val_loader, cfg, device)
        # Restore live training weights.
        model.load_state_dict(prev, strict=True)

        train_avg = {k: running[k] / max(1, n) for k in running}
        wall = time.time() - t_start
        writer.writerow(
            [
                epoch, global_step, optim.param_groups[0]["lr"],
                train_avg["loss"], train_avg["loss_cls"],
                train_avg["loss_gmm"], train_avg["loss_pen"],
                val["loss"], val["loss_cls"], val["loss_gmm"],
                val["loss_pen"], val["acc_top1_final"], wall,
            ]
        )
        csv_file.flush()
        print(
            f"  epoch {epoch}: train_loss={train_avg['loss']:.4f}    "
            f"val_loss={val['loss']:.4f}    val_top1={val['acc_top1_final']:.4f}"
        )

        # Best-checkpoint + early stopping
        improved = val["loss"] < best_val - 1e-4
        if improved:
            best_val = val["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optim": optim.state_dict(),
                    "config": cfg,
                    "config_hash": config_hash(cfg),
                    "epoch": epoch,
                    "val": val,
                    "git_sha": env_summary()["git_sha"],
                },
                ckpt_dir / "sketchmamba_best.pt",
            )
        else:
            bad_epochs += 1
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optim": optim.state_dict(),
                "config": cfg,
                "config_hash": config_hash(cfg),
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "bad_epochs": bad_epochs,
                "val": val,
            },
            ckpt_dir / "sketchmamba_last.pt",
        )
        if bad_epochs >= cfg["train"]["early_stop_patience"]:
            print(f"Early stopping at epoch {epoch} (no improvement for {bad_epochs} epochs).")
            break

    csv_file.close()
    return ckpt_dir / "sketchmamba_best.pt"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    best = train(cfg)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
