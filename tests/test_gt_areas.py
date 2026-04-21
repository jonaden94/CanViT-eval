"""Correctness tests for the bincount-vectorized gt_areas builder."""

import numpy as np
import pytest

from canvit_eval.tasks.ade20k_obj.gt_areas import (
    MAX_CLASS_IDX,
    MIN_CLASS_IDX,
    _BINCOUNT_BINS,
    compute_class_area,
)


def _bincount_areas(arr: np.ndarray) -> dict[int, float]:
    """What build_image_class_dataframe computes, isolated from the file I/O."""
    counts = np.bincount(arr.ravel(), minlength=_BINCOUNT_BINS)
    inv_size = 1.0 / arr.size
    return {c: float(counts[c] * inv_size) for c in range(MIN_CLASS_IDX, MAX_CLASS_IDX + 1) if counts[c] > 0}


def _per_class_areas(arr: np.ndarray) -> dict[int, float]:
    """Reference: one `(arr == c).sum()` per class — Sabrina's original."""
    out: dict[int, float] = {}
    for c in np.unique(arr).tolist():
        if c < MIN_CLASS_IDX or c > MAX_CLASS_IDX:
            continue
        out[int(c)] = compute_class_area(arr, c)
    return out


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_bincount_matches_per_class(seed):
    """Vectorised path matches per-class passes to float precision on random
    masks drawn from the same pixel-value distribution as ADE20K PNGs."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, MAX_CLASS_IDX + 1, size=(512, 512), dtype=np.uint8)
    bc = _bincount_areas(arr)
    ref = _per_class_areas(arr)
    assert bc.keys() == ref.keys(), (sorted(bc), sorted(ref))
    for c in ref:
        assert bc[c] == pytest.approx(ref[c], abs=1e-12), (c, bc[c], ref[c])


def test_skips_background_and_oob():
    """Background (class 0) + out-of-range (>150) pixels must not appear."""
    arr = np.zeros((16, 16), dtype=np.uint8)
    arr[0:4, 0:4] = 0          # background
    arr[4:8, 0:4] = 3          # valid class 3
    arr[8:12, 0:4] = 160       # out-of-range (ADE20K is 1..150)
    arr[12:, 0:4] = MAX_CLASS_IDX + 1  # 151 → out-of-range
    bc = _bincount_areas(arr)
    # Only class 3 should appear.
    assert set(bc) == {3}
    assert bc[3] == pytest.approx(16 / 256)


def test_all_background():
    """Pure-background mask → empty output."""
    arr = np.zeros((8, 8), dtype=np.uint8)
    assert _bincount_areas(arr) == {}


def test_covers_full_class_range():
    """A mask containing every valid class → N_CLASSES rows."""
    # 150 classes spread over an array of at least 150 pixels.
    arr = np.arange(1, MAX_CLASS_IDX + 1, dtype=np.uint8).reshape(15, 10)
    bc = _bincount_areas(arr)
    assert len(bc) == MAX_CLASS_IDX - MIN_CLASS_IDX + 1 == 150
    for c in range(MIN_CLASS_IDX, MAX_CLASS_IDX + 1):
        assert bc[c] == pytest.approx(1 / arr.size)
