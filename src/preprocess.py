"""One-time preprocessing: pack all 58 Sketch-RNN NPZ files into a single
memory-mappable binary plus an index, and compute global Δx / Δy standard
deviations on the training split.

Outputs (all paths relative to repo root):
    data/packed/data.bin             int16, shape = (total_points, 3)
    data/packed/index.npy            structured array, one row per drawing
    data/packed/meta.json            classes, split sizes, normalization stats
    exported/sketchmamba_v1/norm_stats.npz
    exported/sketchmamba_v1/label_map.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

SPLITS = ("train", "valid", "test")
SPLIT_TO_IDX = {s: i for i, s in enumerate(SPLITS)}

# Stroke-3 clipping (Sketch-RNN convention)
DELTA_CLIP = 1000


INDEX_DTYPE = np.dtype(
    [
        ("label", np.int16),
        ("split", np.int8),
        ("offset", np.int64),
        ("length", np.int32),
    ]
)


def list_classes(npz_dir: Path) -> list[str]:
    classes = sorted(p.stem for p in npz_dir.glob("*.npz"))
    if not classes:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")
    return classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="sketchrnn_npz", type=Path)
    ap.add_argument("--out", default="data/packed", type=Path)
    ap.add_argument(
        "--export_dir",
        default="exported/sketchmamba_v1",
        type=Path,
        help="Where to write norm_stats.npz and label_map.json for the webapp.",
    )
    args = ap.parse_args()

    src: Path = args.src
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    args.export_dir.mkdir(parents=True, exist_ok=True)

    classes = list_classes(src)
    print(f"Found {len(classes)} classes in {src}")

    # ----- Pass 1: count points and drawings -----
    total_points = 0
    total_drawings = 0
    per_split_counts = {s: 0 for s in SPLITS}
    class_split_sizes: dict[str, dict[str, int]] = {}

    for cname in tqdm(classes, desc="Pass 1/2 (counting)"):
        with np.load(src / f"{cname}.npz", allow_pickle=True, encoding="latin1") as z:
            class_split_sizes[cname] = {}
            for split in SPLITS:
                arr = z[split]
                class_split_sizes[cname][split] = len(arr)
                per_split_counts[split] += len(arr)
                total_drawings += len(arr)
                for d in arr:
                    total_points += int(d.shape[0])

    print(
        f"  total drawings = {total_drawings:,}    "
        f"total points = {total_points:,}    "
        f"avg pts/drawing = {total_points / total_drawings:.1f}"
    )
    for s in SPLITS:
        print(f"  {s:>5}: {per_split_counts[s]:,} drawings")

    # ----- Pass 2: write packed binary + index, compute train Δ-std -----
    data_path = out / "data.bin"
    index = np.zeros(total_drawings, dtype=INDEX_DTYPE)

    # Welford-style online stats on train deltas (after clipping).
    n_seen = 0
    mean = np.zeros(2, dtype=np.float64)
    M2 = np.zeros(2, dtype=np.float64)

    point_cursor = 0
    draw_cursor = 0

    # Open data.bin for sequential write.
    with open(data_path, "wb") as fout:
        for label_idx, cname in enumerate(tqdm(classes, desc="Pass 2/2 (packing)")):
            with np.load(src / f"{cname}.npz", allow_pickle=True, encoding="latin1") as z:
                for split in SPLITS:
                    arr = z[split]
                    split_idx = SPLIT_TO_IDX[split]
                    for d in arr:
                        # d : int16 array of shape (L, 3) = (Δx, Δy, pen_up)
                        d = np.asarray(d, dtype=np.int16)
                        if d.ndim != 2 or d.shape[1] != 3:
                            raise ValueError(
                                f"Bad shape {d.shape} in class={cname} split={split}"
                            )
                        # Clip Δ to the Sketch-RNN range.
                        np.clip(d[:, :2], -DELTA_CLIP, DELTA_CLIP, out=d[:, :2])
                        L = int(d.shape[0])

                        fout.write(d.tobytes(order="C"))
                        index[draw_cursor] = (label_idx, split_idx, point_cursor, L)

                        # Update training-split running stats.
                        if split == "train":
                            x = d[:, :2].astype(np.float64)
                            n_old = n_seen
                            n_seen += x.shape[0]
                            delta = x - mean
                            mean += delta.sum(axis=0) / n_seen
                            delta2 = x - mean
                            M2 += (delta * delta2).sum(axis=0)

                        point_cursor += L
                        draw_cursor += 1

    assert draw_cursor == total_drawings
    assert point_cursor == total_points

    # Final std (population variance is fine here; n is huge).
    var = M2 / max(n_seen, 1)
    dx_std, dy_std = float(np.sqrt(var[0])), float(np.sqrt(var[1]))
    print(f"  train Δx std = {dx_std:.4f}    Δy std = {dy_std:.4f}")

    np.save(out / "index.npy", index)

    meta = {
        "classes": classes,
        "num_classes": len(classes),
        "splits": list(SPLITS),
        "split_counts": per_split_counts,
        "class_split_sizes": class_split_sizes,
        "total_points": int(total_points),
        "total_drawings": int(total_drawings),
        "dx_std": dx_std,
        "dy_std": dy_std,
        "delta_clip": DELTA_CLIP,
        "data_dtype": "int16",
        "data_shape_per_point": 3,
        "index_dtype": [list(t) for t in INDEX_DTYPE.descr],
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Webapp artefacts
    np.savez(
        args.export_dir / "norm_stats.npz",
        dx_std=np.float32(dx_std),
        dy_std=np.float32(dy_std),
        delta_clip=np.int32(DELTA_CLIP),
    )
    with open(args.export_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({str(i): c for i, c in enumerate(classes)}, f, indent=2)

    print(f"Wrote {data_path} ({data_path.stat().st_size / 1e9:.2f} GB)")
    print(f"Wrote {out / 'index.npy'} (rows={total_drawings:,})")
    print(f"Wrote {out / 'meta.json'}")


if __name__ == "__main__":
    main()
