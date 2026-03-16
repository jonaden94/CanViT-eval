"""Evaluation metrics: IoU accumulation for segmentation."""

import torch
from torch import Tensor


class IoUAccumulator:
    """Global mIoU: sum intersection/union across all images, then average over classes.

    Uses histc-based counting (DINOv3 approach) — no GPU sync until compute().
    """

    def __init__(self, num_classes: int, ignore_index: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.intersection = torch.zeros(num_classes, device=device)
        self.union = torch.zeros(num_classes, device=device)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Accumulate from [B, H, W] predictions and targets."""
        assert preds.shape == targets.shape
        for i in range(preds.shape[0]):
            self._update_single(preds[i], targets[i])

    def _update_single(self, preds: Tensor, targets: Tensor) -> None:
        mask = targets != self.ignore_index
        p, t = preds[mask], targets[mask]
        for cls in range(self.num_classes):
            p_cls = p == cls
            t_cls = t == cls
            self.intersection[cls] += (p_cls & t_cls).sum()
            self.union[cls] += (p_cls | t_cls).sum()

    def compute(self) -> float:
        """Global mIoU. GPU sync happens here."""
        iou = self.intersection / (self.union + 1e-8)
        valid = self.union > 0
        return iou[valid].mean().item() if valid.any() else 0.0
