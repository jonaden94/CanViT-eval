"""Per-(image, class[, timestep]) IoU for DINOv3 and CanViT on ADE20K validation set.

DINOv3: loads pre-extracted features.pt, runs probe, saves per-(image, class) parquet.
CanViT: runs episodes on the fly with EntropyC2F policy, saves per-(image, class, timestep)
        parquet (timestep 0-indexed: 0..n_timesteps-1).

Usage:
    # DINOv3 (pre-computed features):
    uv run python canvit_eval/tasks/ade20k_obj/export_dataframe.py dinov3

    # CanViT (on-the-fly inference):
    uv run python canvit_eval/tasks/ade20k_obj/export_dataframe.py canvit \
        --canvas-resolutions 8 32 64
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tyro
from canvit_pytorch import CanViTForSemanticSegmentation, SegmentationProbe
from canvit_specialize.datasets.ade20k import (
    IGNORE_LABEL, NUM_CLASSES, ADE20kDataset, ResizeMode, make_val_transforms,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_eval.config import EpisodeConfig, ade20k_root
from canvit_eval.runner import eval_batches
from canvit_eval.tasks.base import TaskConfig

log = logging.getLogger(__name__)

DV3_PROBE_REPO_TEMPLATE = "canvit/probe-ade20k-40k-dv3b-{resolution}px"
CANVIT_PROBE_REPOS: dict[int, str] = {
    8:  "canvit/probe-ade20k-40k-s512-c8-in21k",
    9:  "canvit/probe-ade20k-40k-s512-c9-in21k",
    10: "canvit/probe-ade20k-40k-s512-c10-in21k",
    12: "canvit/probe-ade20k-40k-s512-c12-in21k",
    16: "canvit/probe-ade20k-40k-s512-c16-in21k",
    24: "canvit/probe-ade20k-40k-s512-c24-in21k",
    32: "canvit/probe-ade20k-40k-s512-c32-in21k",
    64: "canvit/probe-ade20k-40k-s1024-c64-in21k",
}

DEFAULT_EXPORTS_DIR = Path(__file__).parents[4] / "explore-ade20k" / "outputs"
DEFAULT_AREA_PARQUET = DEFAULT_EXPORTS_DIR / "ade20k_df_flat.parquet"


@dataclass(kw_only=True)
class BaseConfig(TaskConfig):
    scene_size: int = 512
    resize_mode: ResizeMode = "squish"
    """How masks are fit to scene_size: 'squish' or 'center_crop'."""
    area_parquet: Path = DEFAULT_AREA_PARQUET


@dataclass(kw_only=True)
class DINOv3Config(BaseConfig):
    """DINOv3: load pre-extracted features.pt and run probe per resolution."""
    output: Path = Path("output/dv3_ade20k_per_image.parquet")
    exports_dir: Path = DEFAULT_EXPORTS_DIR
    resolutions: list[int] = field(default_factory=lambda: [128, 144, 160, 192, 256, 384, 512])

    def run(self) -> Path:
        return run_dinov3(self)


@dataclass(kw_only=True)
class CanViTConfig(BaseConfig):
    """CanViT: on-the-fly episode with EntropyC2F, multiple canvas resolutions."""
    output: Path = Path("output/canvit_ade20k_per_image.parquet")
    canvas_resolutions: list[int]
    """Canvas resolutions to process, e.g. [8 16 32 64]. Each appends to output parquet."""
    model_repo: str = field(default_factory=lambda: EpisodeConfig().model_repo)
    glimpse_resolution: int = 128
    n_timesteps: int = 21
    """Episode length. Timesteps 0..n_timesteps-1 are stored (0-indexed)."""
    ade20k_root_path: Path = field(default_factory=ade20k_root)

    def run(self) -> Path:
        return run_canvit(self)


# ── Shared utilities ─────────────────────────────────────────────────────────


def _get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _per_image_iou(
    pred: torch.Tensor,  # [H, W] int64, 0-indexed
    mask: torch.Tensor,  # [H, W] int64, 255=ignore
    C: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Histc confusion matrix trick → (inter, union, gt_area), each [C]."""
    valid = mask != IGNORE_LABEL
    p, t_gt = pred[valid], mask[valid]
    cm = torch.histc((p * C + t_gt).float(), bins=C * C, min=0, max=C * C - 1).reshape(C, C)
    inter = cm.diag()
    union = cm.sum(1) + cm.sum(0) - inter
    return inter, union, cm.sum(0)  # gt_area = col sums


def _load_area_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["image_idx", "class_idx", "area", "class_name"])


def _miou_sanity(inter_t: torch.Tensor, union_t: torch.Tensor, label: str) -> None:
    """Print global mIoU from summed intersection/union tensors [N, C]."""
    total_inter = inter_t.sum(0)
    total_union = union_t.sum(0)
    valid = total_union > 0
    miou = (total_inter[valid] / total_union[valid]).mean().item()
    log.info("  sanity mIoU @ %s = %.2f%%", label, 100 * miou)


def _to_long_df(
    inter: torch.Tensor,  # [N, C]
    union: torch.Tensor,  # [N, C]
    gt_area: torch.Tensor,  # [N, C]
    extra_cols: dict,
    mask_resolution: int | None = None,
    resize_mode: str | None = None,
) -> pd.DataFrame:
    """Vectorised: one row per (image, class) where union > 0.

    CanViT rows store raw int64 pixel counts (inter_px, union_px, gt_area_px)
    plus provenance columns (mask_resolution, resize_mode) so rows from
    different runs remain interpretable after concatenation.
    DINOv3 rows (mask_resolution=None) store derived iou as float32.
    """
    img_idx, c0 = (union > 0).nonzero(as_tuple=True)
    d: dict = {
        "image_idx": img_idx.numpy(),
        "class_idx": (c0 + 1).numpy(),  # 0→1-indexed to match area parquet
        **{k: v for k, v in extra_cols.items()},
        "gt_area_px": gt_area[img_idx, c0].numpy().astype(np.int64),
    }
    if mask_resolution is not None:
        # CanViT: store raw counts + provenance; iou is derived on read
        d["inter_px"] = inter[img_idx, c0].numpy().astype(np.int64)
        d["union_px"] = union[img_idx, c0].numpy().astype(np.int64)
        d["mask_resolution"] = mask_resolution
        d["resize_mode"] = resize_mode
    else:
        # DINOv3: keep derived iou (mask resolution fixed at 512 by features.pt)
        d["iou"] = (inter / (union + 1e-8))[img_idx, c0].numpy().astype(np.float32)
    return pd.DataFrame(d)


# ── DINOv3 ───────────────────────────────────────────────────────────────────


@torch.inference_mode()
def run_dinov3(cfg: DINOv3Config) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    device = _get_device(cfg.device)
    log.info("DINOv3  device=%s  resolutions=%s", device, cfg.resolutions)

    area_df = _load_area_df(cfg.area_parquet)
    frames = []

    for res in cfg.resolutions:
        features_path = cfg.exports_dir / f"{res}px" / "features.pt"
        if not features_path.exists():
            raise FileNotFoundError(features_path)

        data = torch.load(features_path, map_location="cpu", weights_only=False)
        feats_all: torch.Tensor = data["feats"]   # [N, grid*grid, D]
        masks_all: torch.Tensor = data["masks"]   # [N, H, W]
        grid: int = data["grid"]
        mask_res: int = data["scene_size"]
        N = feats_all.shape[0]

        probe = SegmentationProbe.from_pretrained(
            DV3_PROBE_REPO_TEMPLATE.format(resolution=res)
        ).to(device).eval()
        log.info("  %dpx  N=%d  grid=%d", res, N, grid)

        C = NUM_CLASSES
        inter_all = torch.zeros(N, C)
        union_all = torch.zeros(N, C)
        gt_area_all = torch.zeros(N, C)

        for start in tqdm(range(0, N, cfg.batch_size), desc=f"{res}px", leave=False):
            end = min(start + cfg.batch_size, N)
            feats = feats_all[start:end].to(device)
            masks = masks_all[start:end].to(device).long()
            B = feats.shape[0]
            logits = probe(feats.view(B, grid, grid, -1).float())
            if logits.shape[-1] != mask_res:
                logits = F.interpolate(logits, size=(mask_res, mask_res), mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1)
            for i in range(B):
                inter, union, gt_area = _per_image_iou(preds[i], masks[i], C)
                inter_all[start + i] = inter.cpu()
                union_all[start + i] = union.cpu()
                gt_area_all[start + i] = gt_area.cpu()

        _miou_sanity(inter_all, union_all, f"{res}px")
        df = _to_long_df(inter_all, union_all, gt_area_all, {"resolution": res},
                         mask_resolution=mask_res,
                         resize_mode=data.get("resize_mode", cfg.resize_mode))
        frames.append(df)
        log.info("  %dpx: %d (image, class) pairs", res, len(df))

    combined = pd.concat(frames, ignore_index=True)
    merged = combined.merge(area_df, on=["image_idx", "class_idx"], how="left")
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cfg.output, index=False)
    log.info("Saved %s  (%d rows, %.1f MB)", cfg.output, len(merged), cfg.output.stat().st_size / 1e6)
    return cfg.output


# ── CanViT ────────────────────────────────────────────────────────────────────


@torch.inference_mode()
def _run_one_canvas(canvas_grid: int, cfg: CanViTConfig, device: torch.device) -> pd.DataFrame:
    """Run CanViT episode for one canvas_resolution; return long-form DataFrame."""
    probe_repo = CANVIT_PROBE_REPOS[canvas_grid]
    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
        pretrained_repo=cfg.model_repo,
        probe_repo=probe_repo,
    ).to(device).eval()
    probe = seg.head
    log.info("  canvas_grid=%d  probe=%s", canvas_grid, probe_repo)

    img_tf, mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    dataset = ADE20kDataset(
        root=cfg.ade20k_root_path, split="validation",
        img_transform=img_tf, mask_transform=mask_tf,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    T, N, C = cfg.n_timesteps, len(dataset), NUM_CLASSES

    inter_all   = torch.zeros(T, N, C)
    union_all   = torch.zeros(T, N, C)
    gt_area_all = torch.zeros(N, C)  # GT area is timestep-invariant

    # EntropyGuidedC2F requires power-of-2 canvas grid; fall back to coarse_to_fine otherwise
    policy = "entropy_coarse_to_fine" if (canvas_grid & (canvas_grid - 1)) == 0 else "coarse_to_fine"
    log.info("  policy=%s", policy)
    episode_cfg = EpisodeConfig(
        model_repo=cfg.model_repo,
        policy=policy,
        n_timesteps=T,
        canvas_grid=canvas_grid,
        glimpse_px=cfg.glimpse_resolution,
    )
    policy_kwargs = {"probe": probe, "get_spatial_fn": seg.canvit.get_spatial} if policy == "entropy_coarse_to_fine" else {}

    img_start = 0
    for br in tqdm(
        eval_batches(
            model=seg.canvit, loader=loader, episode_cfg=episode_cfg,
            canvas_grid=canvas_grid, device=device, amp=cfg.amp,
            policy_kwargs=policy_kwargs,
        ),
        desc=f"c{canvas_grid}", total=len(loader),
    ):
        _, masks = br.batch
        masks_dev = masks.to(device).long()
        B = masks_dev.shape[0]

        for step in br.steps:
            spatial = seg.canvit.get_spatial(step.state.canvas).view(B, canvas_grid, canvas_grid, -1)
            logits = probe(spatial.float())
            if logits.shape[-1] != masks_dev.shape[-1]:
                logits = F.interpolate(logits, size=(cfg.scene_size, cfg.scene_size), mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1)

            for i in range(B):
                inter, union, gt_area = _per_image_iou(preds[i], masks_dev[i], C)
                inter_all[step.t, img_start + i] = inter.cpu()
                union_all[step.t, img_start + i] = union.cpu()
                if step.t == 0:
                    gt_area_all[img_start + i] = gt_area.cpu()

        img_start += B

    _miou_sanity(inter_all[T - 1], union_all[T - 1], f"c{canvas_grid} t={T-1}")

    frames = []
    for t in range(T):
        df = _to_long_df(inter_all[t], union_all[t], gt_area_all, {
            "canvas_resolution": canvas_grid,
            "timestep": t,  # 0-indexed: 0..T-1
        }, mask_resolution=cfg.scene_size, resize_mode=cfg.resize_mode)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def run_canvit(cfg: CanViTConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    device = _get_device(cfg.device)
    log.info("CanViT  device=%s  canvas_resolutions=%s", device, cfg.canvas_resolutions)

    unknown = [c for c in cfg.canvas_resolutions if c not in CANVIT_PROBE_REPOS]
    if unknown:
        raise ValueError(f"No probe repo for canvas_resolution={unknown}. Available: {sorted(CANVIT_PROBE_REPOS)}")

    area_df = _load_area_df(cfg.area_parquet)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    combined = pd.DataFrame()
    for cr in cfg.canvas_resolutions:
        # Reload parquet each iteration so partial results survive failures
        if cfg.output.exists():
            existing = pd.read_parquet(cfg.output)
            kept = existing[existing["canvas_resolution"] != cr]
        else:
            kept = pd.DataFrame()

        df = _run_one_canvas(cr, cfg, device)
        torch.cuda.empty_cache()
        df = df.merge(area_df, on=["image_idx", "class_idx"], how="left")
        combined = pd.concat([kept, df], ignore_index=True) if not kept.empty else df
        combined.to_parquet(cfg.output, index=False)
        log.info("  c%d saved — parquet now %d rows  (%.1f MB)", cr, len(combined), cfg.output.stat().st_size / 1e6)

    # JSON sidecar with metadata
    sidecar = cfg.output.with_suffix(".json")
    meta = {
        "model_repo": cfg.model_repo,
        "canvas_resolutions_in_parquet": sorted(int(c) for c in combined["canvas_resolution"].unique()) if not combined.empty else [],
        "canvas_resolutions_this_run": cfg.canvas_resolutions,
        "n_timesteps": cfg.n_timesteps,
        "scene_size": cfg.scene_size,
        "glimpse_resolution": cfg.glimpse_resolution,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    sidecar.write_text(json.dumps(meta, indent=2))
    log.info("Metadata saved to %s", sidecar)
    return cfg.output


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = tyro.cli(
        Union[
            Annotated[DINOv3Config, tyro.conf.subcommand("dinov3")],
            Annotated[CanViTConfig, tyro.conf.subcommand("canvit")],
        ]
    )
    cmd.run()
