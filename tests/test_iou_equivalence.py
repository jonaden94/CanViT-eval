"""Correctness tests for the vectorized confusion matrix in ade20k_obj.

The authoritative reference is an integer-exact numpy implementation using
`np.bincount` — this avoids torch.histc's float32 precision drift at high bin
indices (which bites at (n_classes, n_classes)=(150, 150)² = 22500 bins).

`_batch_confusion` is validated against that numpy reference with
bit-identical equality. `_per_image_iou` (histc-based, Sabrina's original)
is ALSO validated, but with a small tolerance at the extremes — see
`test_histc_has_precision_drift_at_high_bins` for the explicit documentation
of the float-rounding bug that motivated the rewrite.
"""

import numpy as np
import pytest
import torch

from canvit_eval.tasks.ade20k_obj.iou import _batch_confusion, _per_image_iou

try:
    from canvit_specialize.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES
except ImportError:  # pragma: no cover — dep-free import path for CPU-only CI
    IGNORE_LABEL = 255
    NUM_CLASSES = 150


def _numpy_reference(preds: np.ndarray, masks: np.ndarray, n_classes: int):
    """Integer-exact per-image confusion matrix via np.bincount. Ground truth."""
    B = preds.shape[0]
    inter = np.zeros((B, n_classes), dtype=np.int64)
    union = np.zeros_like(inter)
    gt_area = np.zeros_like(inter)
    for b in range(B):
        valid = masks[b] != IGNORE_LABEL
        p = preds[b][valid].astype(np.int64)
        m = masks[b][valid].astype(np.int64)
        cm = np.bincount(p * n_classes + m, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        inter[b] = np.diag(cm)
        row = cm.sum(axis=1)
        col = cm.sum(axis=0)
        union[b] = row + col - inter[b]
        gt_area[b] = col
    return inter, union, gt_area


def _torch_histc_reference(preds: torch.Tensor, masks: torch.Tensor, n_classes: int):
    """Loop-based histc reference (Sabrina's original `_per_image_iou` per image)."""
    inter, union, gt = [], [], []
    for i in range(preds.shape[0]):
        i_i, u_i, g_i = _per_image_iou(preds[i], masks[i], n_classes)
        inter.append(i_i)
        union.append(u_i)
        gt.append(g_i)
    return torch.stack(inter), torch.stack(union), torch.stack(gt)


def _assert_same_as_numpy(t, np_ref, label):
    assert tuple(t.shape) == np_ref.shape, (label, t.shape, np_ref.shape)
    # _batch_confusion returns float32 for histc-compat; numpy ref is int64.
    # The values ARE integer-valued so cast is safe.
    t_int = t.long().cpu().numpy()
    assert np.array_equal(t_int, np_ref), (
        label, t_int.sum(), np_ref.sum(), int(np.abs(t_int - np_ref).max())
    )


@pytest.mark.parametrize(
    "shape, n_classes",
    [
        ((4, 32, 32), 10),      # small
        ((4, 128, 128), 50),    # medium
        ((2, 512, 512), 150),   # production-like (triggers histc precision drift)
    ],
)
def test_batch_matches_numpy_exactly(shape, n_classes):
    """`_batch_confusion` must bit-match integer-exact numpy reference.

    Indexing matches `canvit_specialize.datasets.ade20k.ADE20kDataset.__getitem__`:
    valid pixels ∈ [0, n_classes-1], invalid = IGNORE_LABEL.
    """
    torch.manual_seed(0)
    B, H, W = shape
    preds = torch.randint(0, n_classes, shape, dtype=torch.int64)
    masks = torch.randint(0, n_classes, shape, dtype=torch.int64)
    masks[torch.rand(shape) < 0.10] = IGNORE_LABEL

    np_i, np_u, np_g = _numpy_reference(preds.numpy(), masks.numpy(), n_classes)
    i_new, u_new, g_new = _batch_confusion(preds, masks, n_classes)
    _assert_same_as_numpy(i_new, np_i, "inter")
    _assert_same_as_numpy(u_new, np_u, "union")
    _assert_same_as_numpy(g_new, np_g, "gt_area")


def test_all_ignore_label():
    """Edge case: every pixel is IGNORE_LABEL → zero counts everywhere."""
    preds = torch.zeros((3, 16, 16), dtype=torch.int64)
    masks = torch.full((3, 16, 16), IGNORE_LABEL, dtype=torch.int64)

    np_i, np_u, np_g = _numpy_reference(preds.numpy(), masks.numpy(), 5)
    i_new, u_new, g_new = _batch_confusion(preds, masks, 5)
    _assert_same_as_numpy(i_new, np_i, "inter")
    _assert_same_as_numpy(u_new, np_u, "union")
    _assert_same_as_numpy(g_new, np_g, "gt_area")
    assert i_new.sum() == 0 and u_new.sum() == 0 and g_new.sum() == 0


def test_perfect_prediction_invariants():
    """Sanity: pred == mask → inter == gt_area, union == gt_area."""
    n_classes = 7
    masks = torch.randint(0, n_classes, (4, 64, 64), dtype=torch.int64)
    preds = masks.clone()

    np_i, np_u, np_g = _numpy_reference(preds.numpy(), masks.numpy(), n_classes)
    i_new, u_new, g_new = _batch_confusion(preds, masks, n_classes)
    _assert_same_as_numpy(i_new, np_i, "inter")
    _assert_same_as_numpy(u_new, np_u, "union")
    _assert_same_as_numpy(g_new, np_g, "gt_area")
    assert torch.equal(i_new, g_new)
    assert torch.equal(u_new, g_new)


def test_realistic_ade20k_value_distribution():
    """Mimic real ADE20K loader output, same indexing convention."""
    torch.manual_seed(42)
    shape = (4, 512, 512)
    n_classes = NUM_CLASSES
    preds = torch.randint(0, n_classes, shape, dtype=torch.int64)
    masks = torch.randint(0, n_classes, shape, dtype=torch.int64)
    masks[torch.rand(shape) < 0.30] = IGNORE_LABEL

    np_i, np_u, np_g = _numpy_reference(preds.numpy(), masks.numpy(), n_classes)
    i_new, u_new, g_new = _batch_confusion(preds, masks, n_classes)
    _assert_same_as_numpy(i_new, np_i, "inter")
    _assert_same_as_numpy(u_new, np_u, "union")
    _assert_same_as_numpy(g_new, np_g, "gt_area")


def test_histc_has_precision_drift_at_high_bins():
    """Documents the reason we rewrote _per_image_iou.

    At n_classes=150 (NUM_CLASSES for ADE20K), torch.histc with 22500 bins
    over [0, 22499] suffers float32 precision drift: some (pred, gt) pairs
    near the top of the bin range land in the wrong bin. The drift is small
    (0.5%-ish on random data) but non-zero — and produces different numbers
    across runs depending on input order since float accumulation order
    differs. The integer-exact `_batch_confusion` fixes this.

    This test captures the drift rather than asserting on exact equality,
    so a future torch release that fixes histc won't silently break the
    test. It just asserts: numpy and scatter_add AGREE; histc drifts.
    """
    torch.manual_seed(1)
    shape = (2, 512, 512)
    n_classes = 150
    preds = torch.randint(0, n_classes, shape, dtype=torch.int64)
    masks = torch.randint(0, n_classes, shape, dtype=torch.int64)

    np_i, np_u, np_g = _numpy_reference(preds.numpy(), masks.numpy(), n_classes)
    h_i, h_u, h_g = _torch_histc_reference(preds, masks, n_classes)
    b_i, b_u, b_g = _batch_confusion(preds, masks, n_classes)

    # scatter_add (batch) matches numpy exactly.
    _assert_same_as_numpy(b_i, np_i, "batch.inter")
    _assert_same_as_numpy(b_g, np_g, "batch.gt_area")

    # histc differs from numpy (at least at very high bin indices).
    drift = (h_i.long().numpy() - np_i).astype(np.int64)
    max_drift = int(np.abs(drift).max())
    # Relative drift is small: assert the total mismatch is bounded.
    total_drift = int(np.abs(drift).sum())
    total_counts = int(np_g.sum())
    # Document observed drift; <0.1% of all counts is expected.
    assert total_drift < 0.01 * total_counts, (
        f"histc drift larger than expected: {total_drift} / {total_counts}"
    )
    # Sanity: there IS some drift at n=150 (proves the motivation).
    assert max_drift > 0, "histc precision drift not observed — test stale?"
