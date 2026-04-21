"""Microbenchmark: `_batch_confusion` vs the per-image `_per_image_iou` loop.

Simulates the hottest inner loop of `_run_one_canvas` with production-scale
inputs (n_classes=150, H=W=512, B=8). Reports wall-clock speedup + per-step
throughput on CUDA if available; falls back to CPU for smoke.

Usage:
    uv run python -m canvit_eval.tasks.ade20k_obj.bench_confusion
"""

import argparse
import time

import torch

from canvit_eval.tasks.ade20k_obj.iou import _batch_confusion, _per_image_iou

try:
    from canvit_specialize.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES
except ImportError:
    IGNORE_LABEL = 255
    NUM_CLASSES = 150


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench_old(preds: torch.Tensor, masks: torch.Tensor, n_classes: int, warmup: int, repeats: int) -> float:
    """Per-image histc loop (Sabrina's original)."""
    device = preds.device
    for _ in range(warmup):
        for i in range(preds.shape[0]):
            _per_image_iou(preds[i], masks[i], n_classes)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        for i in range(preds.shape[0]):
            _per_image_iou(preds[i], masks[i], n_classes)
    _sync(device)
    return (time.perf_counter() - t0) / repeats


def _bench_new(preds: torch.Tensor, masks: torch.Tensor, n_classes: int, warmup: int, repeats: int) -> float:
    """Batched scatter_add (the new default)."""
    device = preds.device
    for _ in range(warmup):
        _batch_confusion(preds, masks, n_classes)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _batch_confusion(preds, masks, n_classes)
    _sync(device)
    return (time.perf_counter() - t0) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8, help="images per step (CanViT c=64 effective batch)")
    parser.add_argument("--hw", type=int, default=512, help="mask side length (scene_size_px)")
    parser.add_argument("--n-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ignore-frac", type=float, default=0.3, help="fraction of IGNORE_LABEL pixels")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device={device}  B={args.batch_size}  H=W={args.hw}  n_classes={args.n_classes}")
    print(f"warmup={args.warmup}  repeats={args.repeats}  ignore_frac={args.ignore_frac}")

    torch.manual_seed(0)
    shape = (args.batch_size, args.hw, args.hw)
    preds = torch.randint(0, args.n_classes, shape, dtype=torch.int64, device=device)
    masks = torch.randint(0, args.n_classes, shape, dtype=torch.int64, device=device)
    masks[torch.rand(shape, device=device) < args.ignore_frac] = IGNORE_LABEL

    old_sec = _bench_old(preds, masks, args.n_classes, args.warmup, args.repeats)
    new_sec = _bench_new(preds, masks, args.n_classes, args.warmup, args.repeats)

    speedup = old_sec / new_sec if new_sec > 0 else float("inf")
    print()
    print(f"  per-image loop:    {old_sec * 1000:.2f} ms/batch   ({old_sec * 1000 / args.batch_size:.3f} ms/img)")
    print(f"  batched scatter:   {new_sec * 1000:.2f} ms/batch   ({new_sec * 1000 / args.batch_size:.3f} ms/img)")
    print(f"  speedup:           {speedup:.1f}×")

    # Per-step equivalent in the full CanViT pipeline:
    # T=21 timesteps × 2000 imgs / B ≈ 250 batches per canvas resolution.
    n_steps_per_canvas = 21 * (2000 / args.batch_size)
    old_canvas_s = old_sec * n_steps_per_canvas
    new_canvas_s = new_sec * n_steps_per_canvas
    saved_per_canvas = old_canvas_s - new_canvas_s
    print()
    print(f"  extrapolated cost per canvas_grid (T=21, 2000 imgs, B={args.batch_size}):")
    print(f"    old: {old_canvas_s:.1f} s  ({old_canvas_s / 60:.1f} min)")
    print(f"    new: {new_canvas_s:.1f} s  ({new_canvas_s / 60:.1f} min)")
    print(f"    saved: {saved_per_canvas:.1f} s / canvas")
    print(f"  across 4 canvases: {saved_per_canvas * 4:.1f} s  ({saved_per_canvas * 4 / 60:.1f} min)")


if __name__ == "__main__":
    main()
