"""ADE20K semantic segmentation evaluation.

Two entry points, matching the two model families we compare:

    run_canvit(...)   — multi-timestep CanViT episode, mIoU per timestep.
    run_dinov3(...)   — single passive forward pass of a DINOv3 teacher, mIoU at t=0.

Both write .pt files with the same schema (`mious: {t0: ..., ...}` + metadata),
so the downstream export pipeline is model-agnostic.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from canvit_pytorch import CanViTForSemanticSegmentation, SegmentationProbe
from canvit_pytorch.probes import SegmentationProbe as SegmentationProbeType
from canvit_pytorch.teacher import load_teacher
from canvit_specialize.datasets.ade20k import (
    IGNORE_LABEL, NUM_CLASSES, ADE20kDataset, ResizeMode, make_val_transforms,
)
from canvit_specialize.metrics import mIoUAccumulator
from canvit_specialize.training.utils import collect_metadata
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_eval.config import EpisodeConfig, TEACHER_REPO, ade20k_root
from canvit_eval.runner import eval_batches

log = logging.getLogger(__name__)


@dataclass
class CanViTConfig:
    """Config for the CanViT ADE20K seg eval. Probe and backbone are pulled
    from HF Hub inside run_canvit(); the caller supplies repo IDs via
    --probe-repo and --episode.model-repo."""
    probe_repo: str
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    ade20k_root: Path = field(default_factory=ade20k_root)
    output: Path = Path("results/ade20k_seg.pt")
    scene_size: int = 512
    resize_mode: ResizeMode = "squish"
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True


@dataclass
class DINOv3Config:
    """Config for the passive DINOv3 baseline. No episode — single forward pass
    at `eval_resolution` px."""
    probe_repo: str
    teacher_repo: str = TEACHER_REPO
    eval_resolution: int = 512
    ade20k_root: Path = field(default_factory=ade20k_root)
    output: Path = Path("results/ade20k_seg.pt")
    scene_size: int = 512
    resize_mode: ResizeMode = "squish"
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True


def _loader(cfg: CanViTConfig | DINOv3Config) -> DataLoader:
    img_tf, mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    dataset = ADE20kDataset(
        root=cfg.ade20k_root, split="validation",
        img_transform=img_tf, mask_transform=mask_tf,
    )
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )


def _update_miou(
    acc: mIoUAccumulator, probe: SegmentationProbeType, features: Tensor, masks: Tensor,
) -> None:
    """Run probe on features, interpolate to mask size if needed, accumulate IoU.
    No host syncs: argmax on GPU, mask already on GPU."""
    logits = probe(features.float())
    if logits.shape[-1] != masks.shape[-1]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    acc.update(logits.argmax(dim=1), masks)


def _save(output: Path, mious: dict[str, float], cfg: object, extra_meta: dict, wall_time: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mious": mious,
        "metadata": {**collect_metadata(cfg), **extra_meta, "wall_time_seconds": wall_time},
    }, output)
    for k, v in mious.items():
        log.info("  %s: %.2f%%", k, 100 * v)
    log.info("Saved to %s (%.1fs)", output, wall_time)


def run_canvit(cfg: CanViTConfig) -> Path:
    """CanViT ADE20K eval: T-step episode, mIoU per timestep.

    At each step, pull canvas state → get_spatial → probe → mIoU. No .item()
    inside the episode loop to avoid GPU syncs — accumulators keep tensors
    on device.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg.device)

    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
        pretrained_repo=cfg.episode.model_repo,
        probe_repo=cfg.probe_repo,
    ).to(device).eval()
    probe = seg.head

    canvas_grid = cfg.episode.canvas_grid or cfg.scene_size // seg.canvit.backbone.patch_size_px
    loader = _loader(cfg)
    T = cfg.episode.n_timesteps

    accs = [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(T)]
    t_start = time.monotonic()
    n_images = 0
    policy_kwargs = {"probe": probe, "get_spatial_fn": seg.canvit.get_spatial} \
                    if cfg.episode.policy == "entropy_coarse_to_fine" else {}

    for br in eval_batches(
        model=seg.canvit, loader=loader, episode_cfg=cfg.episode,
        canvas_grid=canvas_grid, device=device, amp=cfg.amp,
        policy_kwargs=policy_kwargs,
    ):
        _, masks = br.batch
        masks = masks.to(device, non_blocking=True)
        B = masks.shape[0]
        n_images += B
        for step in br.steps:
            spatial = seg.canvit.get_spatial(step.state.canvas).view(B, canvas_grid, canvas_grid, -1)
            _update_miou(accs[step.t], probe, spatial, masks)

    # Single compute() per accumulator at the end — these .item() calls happen
    # exactly T times total (not per batch), one GPU→CPU sync after all work is done.
    mious = {f"t{t}": acc.compute() for t, acc in enumerate(accs)}
    wall = time.monotonic() - t_start
    _save(cfg.output, mious, cfg, {
        "n_images": n_images,
        "canvas_grid": canvas_grid,
        "policy": cfg.episode.policy,
        "n_timesteps": T,
        "model_repo": cfg.episode.model_repo,
        "probe_repo": cfg.probe_repo,
    }, wall)
    return cfg.output


def run_dinov3(cfg: DINOv3Config) -> Path:
    """DINOv3 passive ADE20K eval: single forward pass, mIoU at t=0 only.

    No episode structure — this is the passive-vision baseline. Teacher runs
    at cfg.eval_resolution (the resolution its probe was trained at), NOT at
    cfg.scene_size.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg.device)

    teacher = load_teacher(cfg.teacher_repo, device)
    probe = SegmentationProbe.from_pretrained(cfg.probe_repo).to(device).eval()
    grid = cfg.eval_resolution // 16  # DINOv3 patch size
    loader = _loader(cfg)

    acc = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    t_start = time.monotonic()
    n_images = 0
    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp):
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            B = images.shape[0]
            n_images += B
            resized = F.interpolate(
                images, size=(cfg.eval_resolution, cfg.eval_resolution),
                mode="bilinear", align_corners=False,
            )
            feats = teacher.forward_norm_features(resized).patches.view(B, grid, grid, -1)
            _update_miou(acc, probe, feats, masks)

    mious = {"t0": acc.compute()}
    wall = time.monotonic() - t_start
    _save(cfg.output, mious, cfg, {
        "n_images": n_images,
        "eval_resolution": cfg.eval_resolution,
        "teacher_repo": cfg.teacher_repo,
        "probe_repo": cfg.probe_repo,
    }, wall)
    return cfg.output
