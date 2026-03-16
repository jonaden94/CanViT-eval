"""ADE20K semantic segmentation evaluation.

ONE eval pipeline for ALL models (CanViT, DINOv3, anything).
The only thing that varies is how spatial features are extracted per batch.
Dataset loading, probe application, IoU computation are SHARED.
"""

import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch
from canvit_utils.probes import SegmentationProbe
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_eval.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES, ADE20kDataset, ResizeMode, make_val_transforms
from canvit_eval.metrics import IoUAccumulator
from canvit_eval.utils import collect_metadata

log = logging.getLogger(__name__)


# The core abstraction: given a batch of images, produce spatial features per timestep.
# For CanViT: run episode → extract canvas spatial at each t → list of [B, G, G, D]
# For DINOv3: single forward pass → list of one [B, G, G, D]
# This is ALL that differs between models.
FeatureExtractor = Callable[[Tensor], list[Tensor]]


def _default_ade20k_root() -> Path:
    v = os.environ.get("ADE20K_ROOT")
    assert v is not None, "Set ADE20K_ROOT"
    return Path(v)


@dataclass
class Config:
    """ADE20K segmentation eval config — model-agnostic."""

    probe_repo: str
    ade20k_root: Path = field(default_factory=_default_ade20k_root)
    output: Path = Path("results/ade20k_seg.pt")
    scene_size: int = 512
    resize_mode: ResizeMode = "squish"
    batch_size: int = 32
    num_workers: int = 8
    device: str = "cuda"
    amp: bool = True


@torch.inference_mode()
def evaluate(
    cfg: Config,
    extract_features: FeatureExtractor,
    *,
    metadata: dict | None = None,
) -> Path:
    """Run ADE20K segmentation eval with ANY feature extractor.

    Args:
        cfg: Dataset/probe/hardware config.
        extract_features: Given [B, C, H, W] images, returns list of [B, G, G, D]
            feature maps (one per timestep). For passive models, list has one element.
        metadata: Extra metadata to save (model_repo, policy, etc.).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.device)

    probe = SegmentationProbe.from_pretrained(cfg.probe_repo).to(device).eval()

    img_tf, mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    dataset = ADE20kDataset(cfg.ade20k_root, "validation", img_tf, mask_tf)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    # We don't know T until the first batch — allocate lazily
    iou_per_t: list[IoUAccumulator] | None = None
    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
    t_start = time.monotonic()

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp):
        for images, masks in tqdm(loader, desc="ADE20K seg"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            features_per_t = extract_features(images)

            if iou_per_t is None:
                iou_per_t = [IoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in features_per_t]

            assert len(features_per_t) == len(iou_per_t)

            for t, features in enumerate(features_per_t):
                logits = probe(features.float())  # [B, C, G, G]
                if logits.shape[-1] != masks.shape[-1]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=masks.shape[-2:], mode="bilinear", align_corners=False,
                    )
                iou_per_t[t].update(logits.argmax(dim=1), masks)

    assert iou_per_t is not None, "Empty dataset"
    mious = [iou.compute() for iou in iou_per_t]
    wall_time = time.monotonic() - t_start

    for t, m in enumerate(mious):
        log.info("  t%d: %.2f%%", t, 100 * m)

    results = {
        "mious": {f"t{t}": m for t, m in enumerate(mious)},
        "metadata": {
            **(metadata or {}),
            **collect_metadata(cfg),
            "wall_time_seconds": wall_time,
            "n_images": len(dataset),
        },
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, cfg.output)
    log.info("Saved to %s", cfg.output)
    return cfg.output
