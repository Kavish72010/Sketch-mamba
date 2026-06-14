"""Packed-binary Dataset for SketchMamba.

Loads strokes from `data/packed/{data.bin,index.npy,meta.json}` produced by
`src.preprocess`, converts them on the fly from stroke-3 to stroke-5 and
returns padded mini-batches with a length-bucketed sampler that minimises
wasted FLOPs on padded positions.

Stroke-5 layout per time step:
    [Δx_norm, Δy_norm, p1, p2, p3]
where (p1, p2, p3) one-hot encode {pen-down, pen-up, end-of-sketch}.
A leading zero token (pen-down) is prepended; an end-of-sketch token is
appended.  Sequence length therefore equals (raw_length + 2), capped at
`max_len`.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.preprocess import INDEX_DTYPE, SPLIT_TO_IDX

PEN_DOWN, PEN_UP, PEN_END = 0, 1, 2
STROKE5_DIM = 5


@dataclass
class DataMeta:
    classes: list[str]
    num_classes: int
    dx_std: float
    dy_std: float
    delta_clip: int

    @classmethod
    def load(cls, packed_dir: Path) -> "DataMeta":
        with open(packed_dir / "meta.json", "r", encoding="utf-8") as f:
            m = json.load(f)
        return cls(
            classes=m["classes"],
            num_classes=m["num_classes"],
            dx_std=float(m["dx_std"]),
            dy_std=float(m["dy_std"]),
            delta_clip=int(m["delta_clip"]),
        )


class PackedSketchDataset(Dataset):
    """Map-style dataset backed by an int16 memmap.

    Each item returns a dict:
        x     : float32 tensor (T, 5)   — stroke-5, normalised
        label : int64   scalar          — class index
        length: int64   scalar          — true sequence length T  (T ≤ max_len)
    """

    def __init__(
        self,
        packed_dir: str | Path,
        split: str,
        max_len: int = 200,
        augment: bool = False,
        aug_scale_range: tuple[float, float] = (0.9, 1.1),
        aug_stroke_dropout: float = 0.10,
    ) -> None:
        packed_dir = Path(packed_dir)
        if split not in SPLIT_TO_IDX:
            raise ValueError(f"Unknown split: {split}")

        self.meta = DataMeta.load(packed_dir)
        self.max_len = int(max_len)
        self.augment = bool(augment)
        self.aug_scale_lo, self.aug_scale_hi = aug_scale_range
        self.aug_stroke_dropout = float(aug_stroke_dropout)

        # Memmap the raw points file (read-only, shared across workers).
        self._data_path = packed_dir / "data.bin"
        self._data: np.ndarray | None = None  # lazily opened per-worker

        full_index = np.load(packed_dir / "index.npy")
        if full_index.dtype != INDEX_DTYPE:
            # Reinterpret to ensure correct field names.
            full_index = full_index.astype(INDEX_DTYPE)

        mask = full_index["split"] == SPLIT_TO_IDX[split]
        self.index = full_index[mask].copy()
        self.lengths = self.index["length"].astype(np.int64) + 2  # +start+end tokens
        # Cap reported length for bucketing (we'll truncate at __getitem__).
        self.bucket_lengths = np.minimum(self.lengths, self.max_len)

        self.dx_std = self.meta.dx_std
        self.dy_std = self.meta.dy_std

    # --------------------------------------------------------------------- #
    def _ensure_memmap(self) -> np.ndarray:
        if self._data is None:
            self._data = np.memmap(self._data_path, dtype=np.int16, mode="r").reshape(
                -1, 3
            )
        return self._data

    def __len__(self) -> int:  # noqa: D401
        return self.index.shape[0]

    # --------------------------------------------------------------------- #
    def _stroke3_to_stroke5(self, raw: np.ndarray) -> np.ndarray:
        """raw: int16 (L,3) -> float32 (T,5) with start + end tokens.

        Truncates at self.max_len if the resulting sequence is too long.
        """
        L = int(raw.shape[0])
        # raw[:,2] == 1 means pen-up (end of stroke), 0 means pen-down.
        dx = raw[:, 0].astype(np.float32) / self.dx_std
        dy = raw[:, 1].astype(np.float32) / self.dy_std
        pen_up = raw[:, 2].astype(np.int8)

        # Optional augmentation (train only).
        if self.augment:
            # Random isotropic scale.
            s = random.uniform(self.aug_scale_lo, self.aug_scale_hi)
            dx = dx * s
            dy = dy * s
            # Random stroke (point) dropout — only on pen-down points.
            if self.aug_stroke_dropout > 0.0 and L > 4:
                keep = np.ones(L, dtype=bool)
                cand = np.where(pen_up == 0)[0]
                drop_n = int(round(self.aug_stroke_dropout * cand.size))
                if drop_n > 0:
                    drop = np.random.choice(cand, size=drop_n, replace=False)
                    keep[drop] = False
                    dx, dy, pen_up = dx[keep], dy[keep], pen_up[keep]
                    L = int(keep.sum())

        T = L + 2
        if T > self.max_len:
            # Truncate keeping the start token + as many points as fit + end token.
            L = self.max_len - 2
            dx, dy, pen_up = dx[:L], dy[:L], pen_up[:L]
            T = self.max_len

        out = np.zeros((T, STROKE5_DIM), dtype=np.float32)
        # Start token (t=0): zero deltas, pen-down.
        out[0, 2] = 1.0
        # Body.
        out[1 : 1 + L, 0] = dx
        out[1 : 1 + L, 1] = dy
        # pen state: 0 -> p1 (down), 1 -> p2 (up).
        out[1 : 1 + L, 2] = (pen_up == 0).astype(np.float32)
        out[1 : 1 + L, 3] = (pen_up == 1).astype(np.float32)
        # End-of-sketch token.
        out[1 + L, 4] = 1.0
        return out

    def __getitem__(self, idx: int) -> dict:
        rec = self.index[idx]
        offset = int(rec["offset"])
        length = int(rec["length"])
        label = int(rec["label"])
        data = self._ensure_memmap()
        raw = np.asarray(data[offset : offset + length], dtype=np.int16)
        x = self._stroke3_to_stroke5(raw)
        return {
            "x": torch.from_numpy(x),
            "label": torch.tensor(label, dtype=torch.long),
            "length": torch.tensor(x.shape[0], dtype=torch.long),
        }


# ------------------------------------------------------------------------- #
def collate_pad(batch: list[dict]) -> dict:
    """Pad to the longest sequence in the batch."""
    T = max(int(b["length"]) for b in batch)
    B = len(batch)
    x = torch.zeros(B, T, STROKE5_DIM, dtype=torch.float32)
    # Padded positions are end-of-sketch tokens (consistent with sequence end).
    x[..., 4] = 1.0
    labels = torch.empty(B, dtype=torch.long)
    lengths = torch.empty(B, dtype=torch.long)
    for i, b in enumerate(batch):
        L = int(b["length"])
        x[i, :L] = b["x"]
        labels[i] = b["label"]
        lengths[i] = L
    mask = torch.arange(T)[None, :] < lengths[:, None]
    return {"x": x, "label": labels, "length": lengths, "mask": mask}


# ------------------------------------------------------------------------- #
class LengthBucketedSampler(Sampler[list[int]]):
    """Yields batches of indices whose sequence lengths are similar.

    Implementation: shuffle indices, chunk into "buckets" of
    `bucket_size = batch_size * bucket_size_multiplier`, sort each bucket by
    length, split each bucket into batches, then shuffle the batch order.
    Drops the last partial batch by default to keep step counts deterministic.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        bucket_size_multiplier: int = 50,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.bucket = int(batch_size) * int(bucket_size_multiplier)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        n = self.lengths.size
        if drop_last:
            self._num_batches = n // batch_size
        else:
            self._num_batches = math.ceil(n / batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        n = self.lengths.size
        order = rng.permutation(n) if self.shuffle else np.arange(n)
        batches: list[np.ndarray] = []
        for start in range(0, n, self.bucket):
            chunk = order[start : start + self.bucket]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for bs in range(0, chunk.size, self.batch_size):
                batch = chunk[bs : bs + self.batch_size]
                if self.drop_last and batch.size < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        for b in batches:
            yield b.tolist()

    def __len__(self) -> int:
        return self._num_batches


# ------------------------------------------------------------------------- #
def make_loader(
    packed_dir: str | Path,
    split: str,
    batch_size: int,
    max_len: int = 200,
    num_workers: int = 8,
    bucket_size_multiplier: int = 50,
    augment: bool = False,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    seed: int = 0,
    aug_scale_range: tuple[float, float] = (0.9, 1.1),
    aug_stroke_dropout: float = 0.10,
) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"
    ds = PackedSketchDataset(
        packed_dir=packed_dir,
        split=split,
        max_len=max_len,
        augment=augment and split == "train",
        aug_scale_range=aug_scale_range,
        aug_stroke_dropout=aug_stroke_dropout,
    )
    sampler = LengthBucketedSampler(
        lengths=ds.bucket_lengths,
        batch_size=batch_size,
        bucket_size_multiplier=bucket_size_multiplier,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=collate_pad,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader


# ------------------------------------------------------------------------- #
def stratified_subset_indices(
    labels: np.ndarray, n_total: int, seed: int = 0
) -> np.ndarray:
    """Pick a class-stratified subset of `n_total` indices."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    per_class = max(1, n_total // classes.size)
    out: list[np.ndarray] = []
    for c in classes:
        idx = np.where(labels == c)[0]
        take = min(per_class, idx.size)
        out.append(rng.choice(idx, size=take, replace=False))
    return np.concatenate(out)
