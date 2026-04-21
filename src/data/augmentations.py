"""
SketchMamba - Data Augmentations
Stroke-level augmentations that operate on raw numpy drawings (seq_len, 3).
"""

import numpy as np
from typing import Optional


def permute_strokes(drawing: np.ndarray, prob: float = 0.3) -> np.ndarray:
    """Randomly shuffle the order of strokes within a drawing.

    Preserves the point order *within* each stroke but randomizes the order
    of entire strokes. This makes the model robust to different drawing orders.

    Args:
        drawing: (seq_len, 3) array with columns (dx, dy, pen_state).
                 pen_state=1 marks stroke boundaries (pen-up).
        prob: probability of applying the augmentation.

    Returns:
        Possibly permuted drawing of the same shape.
    """
    if np.random.rand() > prob:
        return drawing

    # Split drawing into individual strokes
    strokes = _split_into_strokes(drawing)

    if len(strokes) <= 1:
        return drawing  # nothing to permute

    # Shuffle stroke order
    perm = np.random.permutation(len(strokes))
    strokes = [strokes[i] for i in perm]

    # Reassemble
    return np.concatenate(strokes, axis=0)


def random_truncate(
    drawing: np.ndarray,
    min_ratio: float = 0.1,
    max_ratio: float = 1.0,
) -> np.ndarray:
    """Truncate a drawing to a random prefix.

    The truncation ratio is sampled uniformly from [min_ratio, max_ratio].
    This simulates partial sketches for progressive classification training.

    Args:
        drawing: (seq_len, 3) array.
        min_ratio: minimum fraction of the sequence to keep.
        max_ratio: maximum fraction (1.0 = keep all).

    Returns:
        Truncated drawing. Always keeps at least 3 points.
    """
    seq_len = drawing.shape[0]
    ratio = np.random.uniform(min_ratio, max_ratio)
    keep = max(3, int(seq_len * ratio))
    return drawing[:keep].copy()


def _split_into_strokes(drawing: np.ndarray):
    """Split a drawing into a list of individual strokes.

    Each stroke is a contiguous sub-array ending at a pen_state=1 boundary
    (or at the end of the sequence).
    """
    pen_states = drawing[:, 2]
    # Indices where pen lifts (pen_state == 1)
    boundaries = np.where(pen_states == 1)[0]

    strokes = []
    start = 0
    for end_idx in boundaries:
        strokes.append(drawing[start : end_idx + 1])
        start = end_idx + 1

    # Handle any trailing points after the last pen-up
    if start < len(drawing):
        strokes.append(drawing[start:])

    return strokes


def make_augment_fn(
    stroke_permute_prob: float = 0.3,
    min_truncate_ratio: float = 1.0,
):
    """Factory that returns a combined augmentation callable.

    Args:
        stroke_permute_prob: probability of shuffling stroke order.
        min_truncate_ratio: minimum truncation ratio (1.0 = no truncation).

    Returns:
        A function: (np.ndarray) -> np.ndarray
    """

    def augment(drawing: np.ndarray) -> np.ndarray:
        drawing = permute_strokes(drawing, prob=stroke_permute_prob)
        if min_truncate_ratio < 1.0:
            drawing = random_truncate(drawing, min_ratio=min_truncate_ratio)
        return drawing

    return augment
