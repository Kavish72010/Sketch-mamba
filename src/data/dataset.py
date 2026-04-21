"""
SketchMamba - QuickDraw Dataset
Loads SketchRNN-format NPZ files for all categories, normalizes offsets,
and provides a PyTorch Dataset + collate function for variable-length batching.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict


CATEGORIES: Optional[List[str]] = None  # populated by discover_categories()


def discover_categories(data_dir: str) -> List[str]:
    """Discover all category names from NPZ files in the data directory.

    If no NPZ files are found directly, automatically checks for a
    'sketchrnn_npz' subdirectory (common QuickDraw layout).
    """
    global CATEGORIES
    cats = sorted([
        f.replace(".npz", "")
        for f in os.listdir(data_dir)
        if f.endswith(".npz")
    ])

    # Auto-detect sketchrnn_npz subdirectory if no NPZ files at top level
    if len(cats) == 0:
        sub = os.path.join(data_dir, "sketchrnn_npz")
        if os.path.isdir(sub):
            print(f"[QuickDrawDataset] No NPZ at {data_dir}, using {sub}")
            cats = sorted([
                f.replace(".npz", "")
                for f in os.listdir(sub)
                if f.endswith(".npz")
            ])
            # Update data_dir globally is not safe, so we won't—
            # but callers should use resolve_data_dir() instead.

    CATEGORIES = cats
    return cats


def resolve_data_dir(data_dir: str) -> str:
    """Resolve the actual directory containing NPZ files.

    If data_dir itself has NPZ files, return it as-is.
    Otherwise check for a 'sketchrnn_npz' subdirectory.
    """
    npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    if len(npz_files) > 0:
        return data_dir
    sub = os.path.join(data_dir, "sketchrnn_npz")
    if os.path.isdir(sub):
        npz_sub = [f for f in os.listdir(sub) if f.endswith(".npz")]
        if len(npz_sub) > 0:
            return sub
    return data_dir


def compute_scale_factor(data_dir: str, categories: List[str]) -> float:
    """Compute a single normalization scalar from training-set offsets.

    Returns the max of (std_dx, std_dy) across all training data so that
    normalizing by this value preserves aspect ratio.
    """
    all_offsets = []
    for cat in categories:
        path = os.path.join(data_dir, f"{cat}.npz")
        data = np.load(path, allow_pickle=True, encoding="latin1")
        for drawing in data["train"]:
            all_offsets.append(drawing[:, :2].astype(np.float64))
    all_offsets = np.concatenate(all_offsets, axis=0)
    std_dx = np.std(all_offsets[:, 0])
    std_dy = np.std(all_offsets[:, 1])
    scale = float(max(std_dx, std_dy))
    if scale < 1e-8:
        scale = 1.0  # safety fallback
    return scale


class QuickDrawDataset(Dataset):
    """PyTorch Dataset for SketchRNN-format QuickDraw data.

    Each sample is a (drawing, label) pair where:
        - drawing: (seq_len, 3) float32 tensor with columns (dx, dy, pen_state)
        - label: integer class index
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        categories: Optional[List[str]] = None,
        scale_factor: float = 1.0,
        max_len: Optional[int] = None,
        augment_fn=None,
    ):
        """
        Args:
            data_dir: path to the directory containing .npz files
            split: one of 'train', 'valid', 'test'
            categories: list of category names (if None, auto-discovered)
            scale_factor: normalization scalar for dx/dy
            max_len: optional maximum sequence length (longer seqs are truncated)
            augment_fn: optional callable(drawing_np) -> drawing_np
        """
        assert split in ("train", "valid", "test"), f"Unknown split: {split}"

        if categories is None:
            categories = discover_categories(data_dir)

        self.categories = categories
        self.cat_to_idx: Dict[str, int] = {c: i for i, c in enumerate(categories)}
        self.scale_factor = scale_factor
        self.max_len = max_len
        self.augment_fn = augment_fn

        # Load all data into memory
        self.drawings: List[np.ndarray] = []
        self.labels: List[int] = []

        for cat in categories:
            path = os.path.join(data_dir, f"{cat}.npz")
            data = np.load(path, allow_pickle=True, encoding="latin1")
            split_data = data[split]
            label_idx = self.cat_to_idx[cat]

            for drawing in split_data:
                self.drawings.append(drawing)
                self.labels.append(label_idx)

        print(
            f"[QuickDrawDataset] Loaded {len(self.drawings)} drawings "
            f"from {len(categories)} categories ({split} split)"
        )

    def __len__(self) -> int:
        return len(self.drawings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        drawing = self.drawings[idx].copy().astype(np.float32)
        label = self.labels[idx]

        # Apply augmentations (operates on numpy array)
        if self.augment_fn is not None:
            drawing = self.augment_fn(drawing)

        # Truncate to max_len if specified
        if self.max_len is not None and drawing.shape[0] > self.max_len:
            drawing = drawing[: self.max_len]

        # Normalize dx, dy (columns 0 and 1); pen_state (column 2) stays as-is
        drawing[:, 0] /= self.scale_factor
        drawing[:, 1] /= self.scale_factor

        return torch.from_numpy(drawing), label


def collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sketch sequences into a padded batch.

    Returns:
        strokes: (B, T_max, 3) float32 - padded stroke sequences
        labels:  (B,) int64 - category labels
        mask:    (B, T_max) bool - True for valid (non-padded) positions
        lengths: (B,) int64 - original sequence lengths
    """
    drawings, labels = zip(*batch)
    lengths = torch.tensor([d.shape[0] for d in drawings], dtype=torch.long)
    max_len = int(lengths.max().item())

    batch_size = len(drawings)
    strokes = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (d, length) in enumerate(zip(drawings, lengths)):
        seq_len = int(length.item())
        strokes[i, :seq_len] = d
        mask[i, :seq_len] = True

    labels = torch.tensor(labels, dtype=torch.long)

    return strokes, labels, mask, lengths
