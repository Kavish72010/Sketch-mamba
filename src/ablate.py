"""Run the full ablation matrix declared in configs/ablations.yaml.

Each entry trains, evaluates, and writes results/ablations/<run_name>/
{metrics.json, figures/, train.csv, sketchmamba_best.pt}.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.utils import deep_merge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Optional list of run names to execute (default: all).")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        ab = yaml.safe_load(f)
    base_path = Path(ab["base_config"])
    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    out_root = Path(ab["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for entry in ab["runs"]:
        name = entry["name"]
        if args.only and name not in args.only:
            continue
        cfg = deep_merge(base, entry["overrides"])
        cfg["run_name"] = f"ablation_{name}"
        # Redirect outputs so train + evaluate land under results/ablations/<name>/.
        cfg["train"]["ckpt_dir"] = str(out_root / name / "checkpoints")
        cfg["train"]["log_dir"] = str(out_root / name / "logs")
        cfg["eval"]["metrics_dir"] = str(out_root / name / "metrics")
        cfg["eval"]["figures_dir"] = str(out_root / name / "figures")

        run_dir = out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        # Resume awareness: skip work that's already done.
        ckpt = Path(cfg["train"]["ckpt_dir"]) / cfg["run_name"] / "sketchmamba_best.pt"
        m_path = Path(cfg["eval"]["metrics_dir"]) / cfg["run_name"] / "metrics.json"
        skip_train = ckpt.exists()
        skip_eval = m_path.exists()

        print(f"\n=== Running ablation: {name} ===")
        if skip_train and skip_eval:
            print(f"  -> already complete (best.pt + metrics.json present); skipping.")
        else:
            if skip_train:
                print(f"  best.pt found at {ckpt}; skipping train, running evaluate only.")
            else:
                # Train (will itself resume from sketchmamba_last.pt if present).
                rc = subprocess.call(
                    [sys.executable, "-m", "src.train", "--config", str(cfg_path)]
                )
                if rc != 0:
                    print(f"  training failed for {name}; skipping eval.")
                    continue
            if not skip_eval:
                rc = subprocess.call(
                    [sys.executable, "-m", "src.evaluate",
                     "--ckpt", str(ckpt), "--config", str(cfg_path)]
                )
                if rc != 0:
                    print(f"  evaluate failed for {name}.")
                    continue
            else:
                print(f"  metrics.json found at {m_path}; skipping evaluate.")

        # Pull the headline metrics into the summary file.
        if m_path.exists():
            m = json.loads(m_path.read_text())
            summary[name] = {
                "auc_progressive": m.get("auc_progressive"),
                "earliness_at_0p9": m.get("earliness_at_0p9"),
                "top1_final": m.get("final", {}).get("top1"),
                "f1_macro": m.get("final", {}).get("f1_macro"),
                "pen_acc": m.get("final", {}).get("pen_acc"),
                "gmm_nll_per_step": m.get("final", {}).get("gmm_nll_per_step"),
                "params": m.get("efficiency", {}).get("params"),
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"  -> {name}: top1={summary[name]['top1_final']:.4f}  "
                  f"auc={summary[name]['auc_progressive']:.4f}")

    # Generate a markdown table of results.
    md = ["| run | params | top1 | auc | earl@0.9 | f1 | pen | nll |",
          "| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|"]
    for k, v in summary.items():
        md.append(
            f"| {k} | {v.get('params','-'):,} | {v.get('top1_final',0):.4f} | "
            f"{v.get('auc_progressive',0):.4f} | {v.get('earliness_at_0p9',0):.2f} | "
            f"{v.get('f1_macro',0):.4f} | {v.get('pen_acc',0):.4f} | "
            f"{v.get('gmm_nll_per_step',0):.4f} |"
        )
    (out_root / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote summary to {summary_path} and {out_root/'summary.md'}")


if __name__ == "__main__":
    main()
