"""Export a self-contained inference bundle for the Gradio webapp.

Contents of `<out>/`:
    config.json       — model hparams + tokenization + class list
    weights.pt        — EMA weights only (state_dict, no pickled classes)
    norm_stats.npz    — global Δx/Δy std (already produced by preprocess)
    label_map.json    — index → class name (already produced by preprocess)
    README.md         — minimal load snippet
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml

from src.data import DataMeta
from src.model import build_model
from src.utils import ensure_dir


LOAD_SNIPPET = '''```python
import json, torch, numpy as np
from src.model import build_model

with open("config.json") as f:
    cfg = json.load(f)
model = build_model(cfg["model"])
sd = torch.load("weights.pt", map_location="cpu")
model.load_state_dict(sd, strict=True)
model.eval()
```'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    state = torch.load(args.ckpt, map_location="cpu")
    ckpt_cfg = state["config"]

    out = ensure_dir(args.out)
    # Save weights (EMA preferred).
    sd = state.get("ema") or state["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    torch.save(sd, out / "weights.pt")

    # Save model + tokenization config (a frozen copy).
    meta = DataMeta.load(Path(ckpt_cfg["data"]["packed_dir"]))
    bundle_cfg = {
        "model": {**ckpt_cfg["model"], "max_len": ckpt_cfg["data"]["max_len"] + 8},
        "data": {
            "max_len": ckpt_cfg["data"]["max_len"],
            "dx_std": meta.dx_std,
            "dy_std": meta.dy_std,
            "delta_clip": meta.delta_clip,
        },
        "classes": meta.classes,
        "config_hash": state.get("config_hash"),
        "git_sha": state.get("git_sha"),
        "epoch": state.get("epoch"),
        "val_metrics": state.get("val"),
    }
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(bundle_cfg, f, indent=2)

    # Copy norm_stats.npz / label_map.json from the export dir produced by preprocess
    # (they live next to config.json by convention).
    src_export = Path("exported/sketchmamba_v1")
    for fname in ("norm_stats.npz", "label_map.json"):
        src = src_export / fname
        if src.exists() and src.resolve() != (out / fname).resolve():
            shutil.copy2(src, out / fname)

    with open(out / "README.md", "w", encoding="utf-8") as f:
        f.write(
            f"# SketchMamba inference bundle\n\n"
            f"Model: {bundle_cfg['model']['type']}  |  "
            f"params (state_dict tensors): {sum(v.numel() for v in sd.values()):,}\n\n"
            f"## Loading\n\n{LOAD_SNIPPET}\n"
        )
    print(f"Wrote bundle to {out}")


if __name__ == "__main__":
    main()
