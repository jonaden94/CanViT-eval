"""Per-(image, class[, timestep]) IoU for DINOv3 and CanViT on ADE20K validation set.

DINOv3: loads pre-extracted features.pt, runs probe, writes DV3_PARQUET.
CanViT: runs episodes on the fly with EntropyC2F policy, writes CANVIT_PARQUET
        with one row per (image, class, timestep).

Usage:
    uv run python -m canvit_eval.tasks.ade20k_obj.dataframe_iou_mask_size dinov3
    uv run python -m canvit_eval.tasks.ade20k_obj.dataframe_iou_mask_size canvit --canvas-resolutions 8 32 64
"""

import gc
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Union

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
from canvit_eval.tasks.ade20k_obj.paths import (
    AREA_PARQUET,
    CANVIT_PARQUET,
    DV3_PARQUET,
    FEATURES_DIR,
    features_path,
)
from canvit_eval.tasks.base import TaskConfig

log = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _canvas_policy(canvas_grid: int) -> str:
    return "entropy_coarse_to_fine" if (canvas_grid & (canvas_grid - 1)) == 0 else "coarse_to_fine"

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


@dataclass(kw_only=True)
class BaseConfig(TaskConfig):
    scene_size_px: int = 512
    resize_mode: ResizeMode = "squish"
    area_parquet: Path = AREA_PARQUET


@dataclass(kw_only=True)
class DINOv3Config(BaseConfig):
    """Load pre-extracted DINOv3 features.pt and run probe per resolution."""
    output: Path = DV3_PARQUET
    exports_dir: Path = FEATURES_DIR
    resolutions_px: list[int] = field(default_factory=lambda: sorted(
        int(p.stem.removesuffix("px_features"))
        for p in FEATURES_DIR.glob("*px_features.pt")
    ))

    def run(self) -> Path:
        return run_dinov3(self)


@dataclass(kw_only=True)
class CanViTConfig(BaseConfig):
    """On-the-fly CanViT episodes (EG-C2F) → per-(image, class, timestep) IoU.

    Parquet columns: image_idx, class_idx, gt_area_px, inter_px, union_px,
    canvas_resolution, timestep, mask_resolution, resize_mode.
    """

    output: Path = CANVIT_PARQUET
    canvas_resolutions: list[int] = field(default_factory=list)
    model_repo: str = field(default_factory=lambda: EpisodeConfig().model_repo)
    glimpse_resolution_px: int = 128
    n_timesteps: int = 21
    ade20k_root_path: Path = field(default_factory=ade20k_root)

    def run(self) -> Path:
        return run_canvit(self)


# ── Shared utilities ─────────────────────────────────────────────────────────


def _per_image_iou(
    pred: torch.Tensor,
    mask: torch.Tensor,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Histc confusion matrix → (inter, union, gt_area), each [n_classes].

    pred: [H, W] int64, 0-indexed class predictions.
    mask: [H, W] int64, 1-indexed GT; IGNORE_LABEL excluded from the confusion.
    """
    valid = mask != IGNORE_LABEL
    p, t_gt = pred[valid], mask[valid]
    cm = torch.histc(
        (p * n_classes + t_gt).float(),
        bins=n_classes * n_classes, min=0, max=n_classes * n_classes - 1,
    ).reshape(n_classes, n_classes)
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
    inter: torch.Tensor,
    union: torch.Tensor,
    gt_area: torch.Tensor,
    extra_cols: dict,
    mask_resolution_px: int | None = None,
    resize_mode: str | None = None,
) -> pd.DataFrame:
    """Vectorised: one row per (image, class) where union > 0.

    inter / union / gt_area are all [N, n_classes].
    CanViT rows (mask_resolution_px set) store raw int64 pixel counts + provenance
    (mask_resolution_px, resize_mode) so rows from different runs remain
    interpretable after concatenation. DINOv3 rows store derived iou as float32
    (mask resolution is fixed at 512 by features.pt).
    """
    assert inter.shape == union.shape == gt_area.shape, (inter.shape, union.shape, gt_area.shape)
    img_idx, c0 = (union > 0).nonzero(as_tuple=True)
    d: dict = {
        "image_idx": img_idx.numpy(),
        "class_idx": (c0 + 1).numpy(),  # 0→1-indexed to match area parquet
        **{k: v for k, v in extra_cols.items()},
        "gt_area_px": gt_area[img_idx, c0].numpy().astype(np.int64),
    }
    if mask_resolution_px is not None:
        d["inter_px"] = inter[img_idx, c0].numpy().astype(np.int64)
        d["union_px"] = union[img_idx, c0].numpy().astype(np.int64)
        d["mask_resolution_px"] = mask_resolution_px
        d["resize_mode"] = resize_mode
    else:
        d["iou"] = (inter / (union + 1e-8))[img_idx, c0].numpy().astype(np.float32)
    return pd.DataFrame(d)


# ── DINOv3 ───────────────────────────────────────────────────────────────────


@torch.inference_mode()
def run_dinov3(cfg: DINOv3Config) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    device = torch.device(cfg.device)
    if not cfg.resolutions_px:
        raise ValueError(
            f"No DINOv3 features found under {cfg.exports_dir}. "
            f"Run export_dv3_features.py first (for each resolution you need)."
        )
    log.info("DINOv3  device=%s  resolutions_px=%s", device, cfg.resolutions_px)
    log.info("area_parquet=%s  output=%s", cfg.area_parquet, cfg.output)

    area_df = _load_area_df(cfg.area_parquet)
    frames = []
    n_classes = NUM_CLASSES

    for res_px in cfg.resolutions_px:
        path = features_path(res_px)
        if not path.exists():
            raise FileNotFoundError(path)

        data = torch.load(path, map_location="cpu", weights_only=False)
        feats_all: torch.Tensor = data["feats"]
        masks_all: torch.Tensor = data["masks"]
        grid: int = data["grid"]
        mask_res_px: int = data.get("scene_size_px", data.get("scene_size"))
        n_images = feats_all.shape[0]
        assert feats_all.ndim == 3 and feats_all.shape[1] == grid * grid, feats_all.shape

        probe = SegmentationProbe.from_pretrained(
            DV3_PROBE_REPO_TEMPLATE.format(resolution=res_px)
        ).to(device).eval()
        log.info("  %dpx  n_images=%d  grid=%d", res_px, n_images, grid)

        inter_all = torch.zeros(n_images, n_classes)
        union_all = torch.zeros(n_images, n_classes)
        gt_area_all = torch.zeros(n_images, n_classes)

        for start in tqdm(range(0, n_images, cfg.batch_size), desc=f"{res_px}px", leave=False):
            end = min(start + cfg.batch_size, n_images)
            feats = feats_all[start:end].to(device)
            masks = masks_all[start:end].to(device).long()
            B = feats.shape[0]
            logits = probe(feats.view(B, grid, grid, -1).float())
            if logits.shape[-1] != mask_res_px:
                logits = F.interpolate(logits, size=(mask_res_px, mask_res_px), mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1)
            for i in range(B):
                inter, union, gt_area = _per_image_iou(preds[i], masks[i], n_classes)
                inter_all[start + i] = inter.cpu()
                union_all[start + i] = union.cpu()
                gt_area_all[start + i] = gt_area.cpu()

        _miou_sanity(inter_all, union_all, f"{res_px}px")
        df = _to_long_df(
            inter_all, union_all, gt_area_all, {"resolution": res_px},
            mask_resolution_px=mask_res_px,
            resize_mode=data.get("resize_mode", cfg.resize_mode),
        )
        frames.append(df)
        log.info("  %dpx: %d (image, class) pairs", res_px, len(df))

    combined = pd.concat(frames, ignore_index=True)
    merged = combined.merge(area_df, on=["image_idx", "class_idx"], how="left")
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cfg.output, index=False)
    log.info("Saved %s  (%d rows, %.1f MB)", cfg.output, len(merged), cfg.output.stat().st_size / 1e6)

    sidecar = cfg.output.with_suffix(".json")
    meta: dict[str, Any] = {
        "resolutions_px": cfg.resolutions_px,
        "probe_repos": {r: DV3_PROBE_REPO_TEMPLATE.format(resolution=r) for r in cfg.resolutions_px},
        "exports_dir": str(cfg.exports_dir),
        "scene_size_px": cfg.scene_size_px,
        "resize_mode": cfg.resize_mode,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    sidecar.write_text(json.dumps(meta, indent=2))
    log.info("Metadata saved to %s", sidecar)
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

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, cfg.resize_mode)
    dataset = ADE20kDataset(
        root=cfg.ade20k_root_path, split="validation",
        img_transform=img_tf, mask_transform=mask_tf,
    )
    # Larger canvas grids would OOM at the default batch size.
    effective_bs = min(cfg.batch_size, 8) if canvas_grid >= 24 else cfg.batch_size
    if effective_bs != cfg.batch_size:
        log.info("  canvas_grid=%d: batch_size %d→%d", canvas_grid, cfg.batch_size, effective_bs)
    loader = DataLoader(
        dataset, batch_size=effective_bs, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    T, n_images, n_classes = cfg.n_timesteps, len(dataset), NUM_CLASSES

    inter_all   = torch.zeros(T, n_images, n_classes)
    union_all   = torch.zeros(T, n_images, n_classes)
    gt_area_all = torch.zeros(n_images, n_classes)  # GT area is timestep-invariant

    # EntropyGuidedC2F requires power-of-2 canvas grid; fall back to coarse_to_fine.
    policy = _canvas_policy(canvas_grid)
    log.info("  policy=%s", policy)
    episode_cfg = EpisodeConfig(
        model_repo=cfg.model_repo,
        policy=policy,
        n_timesteps=T,
        canvas_grid=canvas_grid,
        glimpse_px=cfg.glimpse_resolution_px,
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
            assert logits.ndim == 4 and logits.shape[0] == B, logits.shape
            needs_upsample = logits.shape[-1] != masks_dev.shape[-1]

            for i in range(B):
                # Interpolate one image at a time — full-batch upsample to 512² OOMs at large canvas_grid.
                logit_i = logits[i : i + 1]
                if needs_upsample:
                    logit_i = F.interpolate(logit_i, size=(cfg.scene_size_px, cfg.scene_size_px), mode="bilinear", align_corners=False)
                pred_i = logit_i.argmax(dim=1)[0]
                inter, union, gt_area = _per_image_iou(pred_i, masks_dev[i], n_classes)
                inter_all[step.t, img_start + i] = inter.cpu()
                union_all[step.t, img_start + i] = union.cpu()
                if step.t == 0:
                    gt_area_all[img_start + i] = gt_area.cpu()
            del logits

        img_start += B

    _miou_sanity(inter_all[T - 1], union_all[T - 1], f"c{canvas_grid} t={T-1}")

    frames = []
    for t in range(T):
        df = _to_long_df(inter_all[t], union_all[t], gt_area_all, {
            "canvas_resolution": canvas_grid,
            "timestep": t,
        }, mask_resolution_px=cfg.scene_size_px, resize_mode=cfg.resize_mode)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    # policy_kwargs holds a bound method (seg.canvit.get_spatial) which keeps the entire
    # model graph alive; must be deleted explicitly before gc.collect() can free it.
    del seg, probe, policy_kwargs, loader, dataset, inter_all, union_all, gt_area_all, frames
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def _build_subprocess_cmd(cfg: CanViTConfig, canvas_resolution: int) -> list[str]:
    cmd = [sys.executable, "-m", "canvit_eval.tasks.ade20k_obj.dataframe_iou_mask_size", "canvit"]
    cmd += ["--canvas-resolutions", str(canvas_resolution)]
    cmd += ["--output", str(cfg.output)]
    cmd += ["--model-repo", cfg.model_repo]
    cmd += ["--n-timesteps", str(cfg.n_timesteps)]
    cmd += ["--scene-size-px", str(cfg.scene_size_px)]
    cmd += ["--glimpse-resolution-px", str(cfg.glimpse_resolution_px)]
    cmd += ["--resize-mode", cfg.resize_mode]
    cmd += ["--area-parquet", str(cfg.area_parquet)]
    cmd += ["--ade20k-root-path", str(cfg.ade20k_root_path)]
    cmd += ["--device", cfg.device]
    cmd += ["--batch-size", str(cfg.batch_size)]
    cmd += ["--num-workers", str(cfg.num_workers)]
    if not cfg.amp:
        cmd += ["--no-amp"]
    return cmd


def _write_canvit_sidecar(cfg: CanViTConfig, resolutions_in_parquet: list[int]) -> None:
    sidecar = cfg.output.with_suffix(".json")
    meta: dict[str, Any] = {
        "model_repo": cfg.model_repo,
        "canvas_resolutions_in_parquet": resolutions_in_parquet,
        "probe_repos": {cr: CANVIT_PROBE_REPOS[cr] for cr in resolutions_in_parquet if cr in CANVIT_PROBE_REPOS},
        "policies": {cr: _canvas_policy(cr) for cr in resolutions_in_parquet},
        "n_timesteps": cfg.n_timesteps,
        "scene_size_px": cfg.scene_size_px,
        "resize_mode": cfg.resize_mode,
        "glimpse_resolution_px": cfg.glimpse_resolution_px,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    sidecar.write_text(json.dumps(meta, indent=2))
    log.info("Metadata saved to %s", sidecar)


def run_canvit(cfg: CanViTConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    if not cfg.canvas_resolutions:
        raise ValueError(
            f"Specify --canvas-resolutions (any subset of {sorted(CANVIT_PROBE_REPOS)})"
        )
    unknown = [c for c in cfg.canvas_resolutions if c not in CANVIT_PROBE_REPOS]
    if unknown:
        raise ValueError(f"No probe repo for canvas_resolution={unknown}. Available: {sorted(CANVIT_PROBE_REPOS)}")

    # Multi-canvas: one subprocess per resolution so CUDA memory is fully released
    # between runs (in-process cleanup is best-effort; subprocess is bulletproof).
    if len(cfg.canvas_resolutions) > 1:
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        for i, cr in enumerate(cfg.canvas_resolutions):
            log.info("Subprocess %d/%d: canvas_resolution=%d", i + 1, len(cfg.canvas_resolutions), cr)
            subprocess.run(_build_subprocess_cmd(cfg, cr), check=True)
        existing = pd.read_parquet(cfg.output) if cfg.output.exists() else pd.DataFrame()
        resolutions_in_parquet = sorted(int(c) for c in existing["canvas_resolution"].unique()) if not existing.empty else []
        _write_canvit_sidecar(cfg, resolutions_in_parquet)
        return cfg.output

    # Single canvas: run in-process.
    device = torch.device(cfg.device)
    (cr,) = cfg.canvas_resolutions
    log.info("CanViT  device=%s  canvas_resolution=%d", device, cr)
    log.info("area_parquet=%s  output=%s", cfg.area_parquet, cfg.output)

    area_df = _load_area_df(cfg.area_parquet)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    if cfg.output.exists():
        existing = pd.read_parquet(cfg.output)
        kept = existing[existing["canvas_resolution"] != cr]
    else:
        kept = pd.DataFrame()

    df = _run_one_canvas(cr, cfg, device)
    df = df.merge(area_df, on=["image_idx", "class_idx"], how="left")
    combined = pd.concat([kept, df], ignore_index=True) if not kept.empty else df
    combined.to_parquet(cfg.output, index=False)
    log.info("  c%d saved — parquet now %d rows  (%.1f MB)", cr, len(combined), cfg.output.stat().st_size / 1e6)

    _write_canvit_sidecar(cfg, sorted(int(c) for c in combined["canvas_resolution"].unique()))
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
