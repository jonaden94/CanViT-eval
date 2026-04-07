"""ImageNet-1K dataset loading for evaluation.

Preprocessing: Resize shortest side to target_size + CenterCrop.
This matches the manuscript methodology (evaluation_details.typ).
"""

from pathlib import Path

from canvit_pytorch.preprocess import preprocess
from torchvision.datasets import ImageFolder


def make_in1k_dataset(val_dir: Path, target_size: int) -> ImageFolder:
    """Load IN1K validation with canonical preprocessing."""
    assert val_dir.is_dir(), f"IN1K val dir not found: {val_dir}"
    return ImageFolder(str(val_dir), transform=preprocess(target_size))
