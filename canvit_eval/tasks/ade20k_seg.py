"""ADE20K semantic segmentation: frozen CanViT → probe → global mIoU."""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from canvit_utils.probes import SegmentationProbe
from torch.utils.data import DataLoader

from canvit_eval.config import EpisodeConfig, HardwareConfig
from canvit_eval.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES, ADE20kDataset, ResizeMode, make_val_transforms
from canvit_eval.metrics import IoUAccumulator
from canvit_eval.runner import eval_batches, load_model, resolve_canvas_grid
from canvit_eval.utils import collect_metadata

log = logging.getLogger(__name__)


def _default_ade20k_root() -> Path:
    v = os.environ.get("ADE20K_ROOT")
    assert v is not None, "Set ADE20K_ROOT"
    return Path(v)


@dataclass
class Config:
    probe_repo: str
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    hw: HardwareConfig = field(default_factory=HardwareConfig)
    ade20k_root: Path = field(default_factory=_default_ade20k_root)
    output: Path = Path("results/ade20k_seg.pt")
    scene_size: int = 512
    resize_mode: ResizeMode = "squish"


@torch.inference_mode()
def evaluate(cfg: Config) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_float32_matmul_precision("high")
    device = torch.device(cfg.hw.device)

    model = load_model(cfg.episode.model_repo, device)
    probe = SegmentationProbe.from_pretrained(cfg.probe_repo).to(device).eval()
    assert probe.embed_dim == model.canvas_dim
    canvas_grid = resolve_canvas_grid(cfg.episode, model.backbone.patch_size_px, cfg.scene_size)

    img_tf, mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    dataset = ADE20kDataset(cfg.ade20k_root, "validation", img_tf, mask_tf)
    loader = DataLoader(dataset, batch_size=cfg.hw.batch_size, shuffle=False,
                        num_workers=cfg.hw.num_workers, pin_memory=True)

    T = cfg.episode.n_timesteps
    iou_per_t = [IoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(T)]
    t_start = time.monotonic()

    for br in eval_batches(model=model, loader=loader, episode_cfg=cfg.episode,
                           canvas_grid=canvas_grid, device=device, amp=cfg.hw.amp,
                           policy_kwargs={"probe": probe, "get_spatial_fn": model.get_spatial}):
        masks = br.batch[1].to(device, non_blocking=True)
        B = masks.shape[0]
        for step in br.steps:
            features = model.get_spatial(step.state.canvas).view(B, canvas_grid, canvas_grid, -1)
            preds = probe(features.float()).argmax(dim=1)
            if preds.shape[-1] != masks.shape[-1]:
                preds = torch.nn.functional.interpolate(
                    preds.unsqueeze(1).float(), size=masks.shape[-2:], mode="nearest",
                ).squeeze(1).long()
            iou_per_t[step.t].update(preds, masks)

    mious = [iou.compute() for iou in iou_per_t]
    for t, m in enumerate(mious):
        log.info("  t%d: %.2f%%", t, 100 * m)

    results = {
        "mious": {f"t{t}": m for t, m in enumerate(mious)},
        "metadata": {**collect_metadata(cfg), "wall_time_seconds": time.monotonic() - t_start,
                     "n_images": len(dataset), "canvas_grid": canvas_grid},
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, cfg.output)
    log.info("Saved to %s", cfg.output)
    return cfg.output
