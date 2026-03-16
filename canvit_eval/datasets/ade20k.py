"""ADE20K dataset for evaluation.

Supports two resize modes:
- "squish": Resize to exact square (default, matches manuscript methodology)
- "center_crop": Resize shortest side + CenterCrop (easier, not default)

Results are ONLY comparable across models using the same resize_mode.
"""

from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torchvision import transforms as T

NUM_CLASSES = 150
IGNORE_LABEL = 255

ResizeMode = Literal["center_crop", "squish"]


class ADE20kDataset(torch.utils.data.Dataset):
    """ADE20K-SceneParse150 validation dataset."""

    def __init__(self, root: Path, split: str, transform: T.Compose, mask_transform: T.Compose) -> None:
        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        assert img_dir.is_dir(), f"Image dir not found: {img_dir}"
        assert ann_dir.is_dir(), f"Annotation dir not found: {ann_dir}"

        self.images = sorted(img_dir.glob("*.jpg"))
        self.masks = [ann_dir / f"{p.stem}.png" for p in self.images]
        assert len(self.images) > 0, f"No images found in {img_dir}"
        assert all(m.exists() for m in self.masks), "Missing mask files"

        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        img_t = self.transform(img)
        # Masks: subtract 1 (ADE20K uses 1-indexed classes, 0 = ignore)
        mask_t = self.mask_transform(mask).squeeze(0).long() - 1
        mask_t[mask_t < 0] = IGNORE_LABEL
        return img_t, mask_t


def make_val_transforms(size: int, mode: ResizeMode) -> tuple[T.Compose, T.Compose]:
    """Create image and mask transforms for ADE20K validation.

    Returns (image_transform, mask_transform) — always a pair.
    """
    if mode == "center_crop":
        img_transform = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(),
                                    T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        mask_transform = T.Compose([T.Resize(size, T.InterpolationMode.NEAREST), T.CenterCrop(size),
                                     T.PILToTensor()])
    else:
        img_transform = T.Compose([T.Resize((size, size)), T.ToTensor(),
                                    T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        mask_transform = T.Compose([T.Resize((size, size), T.InterpolationMode.NEAREST), T.PILToTensor()])
    return img_transform, mask_transform
