"""Per-(image, class[, timestep]) IoU for DINOv3 and CanViT on ADE20K validation set.

DINOv3: loads pre-extracted features.pt, runs probe, writes DV3_PARQUET.
CanViT: runs episodes on the fly with EntropyC2F policy, writes CANVIT_PARQUET
        with one row per (image, class, timestep).

Usage:
    uv run python -m canvit_eval.tasks.ade20k_obj.iou dinov3
    uv run python -m canvit_eval.tasks.ade20k_obj.iou canvit --canvas-resolutions 8 32 64
"""

import gc
import json
import logging
import subprocess
import sys
import time
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
from canvit_eval.tasks.ade20k_obj.gt_areas import EXPECTED_N_VAL_IMAGES
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

# Upsample-chunk size: bound peak memory at ~1.25 GB regardless of canvas_grid
# (8 × 150 classes × 512² × 4 B). Larger batches get chunked; smaller stay whole.
_UPSAMPLE_CHUNK = 8

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

    Parquet columns (see _to_long_df): image_idx, class_idx, canvas_resolution,
    timestep, inter_px, union_px, gt_area_px, mask_resolution_px, resize_mode
    — plus area + class_name joined from AREA_PARQUET before write.
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


def _batch_confusion(
    preds: torch.Tensor,
    masks: torch.Tensor,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-image (inter, union, gt_area) for a whole batch in a single kernel.

    preds: [B, H, W] int64, class prediction ∈ [0, n_classes − 1].
    masks: [B, H, W] int64, 0-indexed GT ∈ [0, n_classes − 1] ∪ {IGNORE_LABEL}.
      (`canvit_specialize.datasets.ade20k.ADE20kDataset.__getitem__` emits this
       convention via `.long() - 1` + remap `<0` to IGNORE_LABEL.)

    Returns three [B, n_classes] float32 tensors. float32 is chosen for
    compatibility with `_per_image_iou`'s historical dtype AND because
    per-cell counts max out at H·W = 262144 for 512² masks — well below
    float32's exact-integer limit (2²³ = 8.4M). If you ever call with
    H·W > 2²³, promote the accumulator.

    Correctness invariants (all covered by `tests/test_iou_equivalence.py`):
      • Integer-exact match with numpy.bincount reference — the scatter_add
        on a per-image offset reproduces a per-image np.bincount exactly.
      • `(pair >= 0) & (pair < n_classes²)` is defense-in-depth against out
        -of-bounds predictions / masks; argmax on n_classes-wide logits and
        the canvit_specialize loader both guarantee in-range inputs. The
        filter is cheap and catches bugs that would otherwise silently
        corrupt unrelated bins via scatter_add address wrap-around.
      • scatter_add is deterministic on integer-valued accumulators (the
        sum is associative when all summands are exact integers), even
        though GPU scatter_add is non-deterministic in general.
    """
    B, H, W = preds.shape
    assert masks.shape == preds.shape, (preds.shape, masks.shape)
    # int64 is what argmax returns + what the dataloader emits; documenting the
    # contract via assertion (rather than casting) surfaces accidental misuse.
    assert preds.dtype == torch.int64 and masks.dtype == torch.int64, (preds.dtype, masks.dtype)
    device = preds.device
    C2 = n_classes * n_classes

    pair = preds * n_classes + masks                                     # [B, H, W]
    keep = (masks != IGNORE_LABEL) & (pair >= 0) & (pair < C2)
    img_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(preds)
    flat = (img_idx * C2 + pair)[keep]                                    # [K], int64

    # Materialise the confusion matrix as a flat B·C² buffer (per-image
    # offset = b·C²), then view as [B, C, C]. scatter_add_ handles duplicate
    # indices by summing — the natural semantics for confusion accumulation.
    cm = torch.zeros(B * C2, dtype=torch.float32, device=device)
    cm.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    cm = cm.view(B, n_classes, n_classes)

    inter = cm.diagonal(dim1=1, dim2=2)             # [B, C]
    row_sum = cm.sum(dim=2)                         # [B, C]   (pred rows)
    col_sum = cm.sum(dim=1)                         # [B, C]   (gt cols)
    union = row_sum + col_sum - inter
    return inter, union, col_sum                    # gt_area = col sums


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
    mask_resolution_px: int,
    resize_mode: str,
) -> pd.DataFrame:
    """Vectorised: one row per (image, class) where the class is in the GT.

    inter / union / gt_area are all [N, n_classes]. Filter: `gt_area > 0`
    (class actually present in this image's ground truth). This includes:
      • TP-bearing rows: class is in GT and has nonzero prediction overlap —
        iou = inter/union ∈ (0, 1].
      • FN-only rows: class is in GT but the model missed it entirely —
        iou = 0, area = known.
    And excludes:
      • FP-only rows (class NOT in GT, model predicted it anyway): iou would
        be 0 and area undefined (no GT pixels → no entry in area_df). These
        rows are pure noise for the downstream LOWESS(area, iou) fit and
        would contaminate the smooth via statsmodels' default `missing='none'`
        handling. The confusion counts for FP classes still live in the
        full confusion tensor; they're just not emitted to this long-form.

    Stores raw int64 pixel counts (inter_px, union_px, gt_area_px) so rows
    from different runs remain interpretable after concatenation; the
    downstream consumer derives iou as `inter_px / union_px` on read.
    `mask_resolution_px` + `resize_mode` travel with each row as provenance.
    """
    assert inter.shape == union.shape == gt_area.shape, (inter.shape, union.shape, gt_area.shape)
    img_idx, c0 = (gt_area > 0).nonzero(as_tuple=True)
    return pd.DataFrame({
        "image_idx": img_idx.numpy(),
        "class_idx": (c0 + 1).numpy(),  # 0→1-indexed to match area parquet
        **{k: v for k, v in extra_cols.items()},
        "inter_px": inter[img_idx, c0].numpy().astype(np.int64),
        "union_px": union[img_idx, c0].numpy().astype(np.int64),
        "gt_area_px": gt_area[img_idx, c0].numpy().astype(np.int64),
        "mask_resolution_px": mask_resolution_px,
        "resize_mode": resize_mode,
    })


# ── DINOv3 ───────────────────────────────────────────────────────────────────


@torch.inference_mode()
def run_dinov3(cfg: DINOv3Config) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    if not cfg.resolutions_px:
        raise ValueError(
            f"No DINOv3 features found under {cfg.exports_dir}. "
            f"Run `python -m canvit_eval.tasks.ade20k_obj.export_dv3_features` first."
        )
    for r in cfg.resolutions_px:
        assert r > 0 and isinstance(r, int), r

    device = torch.device(cfg.device)
    log.info("=== DINOv3 per-(image, class) IoU ===")
    log.info("device=%s  resolutions_px=%s  batch_size=%d", device, cfg.resolutions_px, cfg.batch_size)
    log.info("area_parquet=%s", cfg.area_parquet)
    log.info("output=%s", cfg.output)

    area_df = _load_area_df(cfg.area_parquet)
    assert len(area_df) > 0, f"area parquet is empty: {cfg.area_parquet}"

    frames: list[pd.DataFrame] = []
    n_classes = NUM_CLASSES
    teacher_repos_seen: set[str] = set()

    for res_px in cfg.resolutions_px:
        path = features_path(res_px)
        if not path.exists():
            raise FileNotFoundError(path)

        t0 = time.monotonic()
        data = torch.load(path, map_location="cpu", weights_only=False)
        feats_all: torch.Tensor = data["feats"]
        masks_all: torch.Tensor = data["masks"]
        grid: int = data["grid"]
        mask_res_px: int = data.get("scene_size_px", data.get("scene_size"))
        n_images = feats_all.shape[0]
        teacher_repo = data.get("teacher_repo", "unknown")
        teacher_repos_seen.add(teacher_repo)

        assert feats_all.ndim == 3 and feats_all.shape[1] == grid * grid, feats_all.shape
        assert masks_all.shape[0] == n_images and masks_all.shape[-1] == mask_res_px, masks_all.shape
        assert n_images == EXPECTED_N_VAL_IMAGES, (
            f"features.pt has {n_images} images, ADE20K val should have {EXPECTED_N_VAL_IMAGES}"
        )
        assert isinstance(mask_res_px, int) and mask_res_px > 0, mask_res_px

        probe_repo = DV3_PROBE_REPO_TEMPLATE.format(resolution=res_px)
        probe = SegmentationProbe.from_pretrained(probe_repo).to(device).eval()
        log.info("  %dpx: features=%s grid=%d mask_res=%d teacher=%s probe=%s",
                 res_px, path.name, grid, mask_res_px, teacher_repo, probe_repo)

        inter_all = torch.zeros(n_images, n_classes)
        union_all = torch.zeros(n_images, n_classes)
        gt_area_all = torch.zeros(n_images, n_classes)

        for start in tqdm(range(0, n_images, cfg.batch_size), desc=f"{res_px}px", leave=False):
            end = min(start + cfg.batch_size, n_images)
            feats = feats_all[start:end].to(device)
            masks = masks_all[start:end].to(device).long()
            B = feats.shape[0]
            logits = probe(feats.view(B, grid, grid, -1).float())
            assert logits.ndim == 4 and logits.shape[0] == B and logits.shape[1] == n_classes, logits.shape
            if logits.shape[-1] != mask_res_px:
                logits = F.interpolate(logits, size=(mask_res_px, mask_res_px), mode="bilinear", align_corners=False)
            preds = logits.argmax(dim=1)
            inter, union, gt_area = _batch_confusion(preds, masks, n_classes)
            inter_all[start:end] = inter.cpu()
            union_all[start:end] = union.cpu()
            gt_area_all[start:end] = gt_area.cpu()

        _miou_sanity(inter_all, union_all, f"{res_px}px")
        df = _to_long_df(
            inter_all, union_all, gt_area_all, {"resolution": res_px},
            mask_resolution_px=mask_res_px,
            resize_mode=data.get("resize_mode", cfg.resize_mode),
        )
        assert len(df) > 0, f"empty dataframe at {res_px}px"
        frames.append(df)
        log.info("  %dpx: %d (image, class) pairs in %.1fs", res_px, len(df), time.monotonic() - t0)

    combined = pd.concat(frames, ignore_index=True)
    merged = combined.merge(area_df, on=["image_idx", "class_idx"], how="left")
    assert len(merged) == len(combined), (len(merged), len(combined))
    # gt_area > 0 filter in _to_long_df guarantees every emitted row has a
    # matching entry in area_df (same (image_idx, class_idx) condition).
    assert not merged["area"].isna().any(), "area merge left NaN rows — area parquet incomplete or out of sync?"
    assert set(merged.columns) >= {"image_idx", "class_idx", "resolution", "inter_px", "union_px", "gt_area_px", "area", "class_name"}, merged.columns

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cfg.output, index=False)
    log.info("saved %s  (%d rows, %.1f MB)", cfg.output, len(merged), cfg.output.stat().st_size / 1e6)

    sidecar = cfg.output.with_suffix(".json")
    meta: dict[str, Any] = {
        "resolutions_px": cfg.resolutions_px,
        "teacher_repos": sorted(teacher_repos_seen),
        "probe_repos": {r: DV3_PROBE_REPO_TEMPLATE.format(resolution=r) for r in cfg.resolutions_px},
        "exports_dir": str(cfg.exports_dir),
        "scene_size_px": cfg.scene_size_px,
        "resize_mode": cfg.resize_mode,
        "n_rows": len(merged),
        "n_classes_with_data": int(merged["class_idx"].nunique()),
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    sidecar.write_text(json.dumps(meta, indent=2))
    log.info("sidecar → %s", sidecar)
    return cfg.output


# ── CanViT ────────────────────────────────────────────────────────────────────


@torch.inference_mode()
def _run_one_canvas(canvas_grid: int, cfg: CanViTConfig, device: torch.device) -> pd.DataFrame:
    """Run CanViT episode for one canvas_resolution; return long-form DataFrame."""
    assert canvas_grid in CANVIT_PROBE_REPOS, (canvas_grid, sorted(CANVIT_PROBE_REPOS))
    t0 = time.monotonic()

    probe_repo = CANVIT_PROBE_REPOS[canvas_grid]
    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
        pretrained_repo=cfg.model_repo,
        probe_repo=probe_repo,
    ).to(device).eval()
    probe = seg.head

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, cfg.resize_mode)
    dataset = ADE20kDataset(
        root=cfg.ade20k_root_path, split="validation",
        img_transform=img_tf, mask_transform=mask_tf,
    )
    T, n_images, n_classes = cfg.n_timesteps, len(dataset), NUM_CLASSES
    assert T > 0, T
    assert n_images == EXPECTED_N_VAL_IMAGES, (n_images, EXPECTED_N_VAL_IMAGES)

    # Larger canvas grids would OOM at the default batch size.
    effective_bs = min(cfg.batch_size, 8) if canvas_grid >= 24 else cfg.batch_size
    loader = DataLoader(
        dataset, batch_size=effective_bs, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # EntropyGuidedC2F requires power-of-2 canvas grid; fall back to coarse_to_fine.
    policy = _canvas_policy(canvas_grid)
    episode_cfg = EpisodeConfig(
        model_repo=cfg.model_repo,
        policy=policy,
        n_timesteps=T,
        canvas_grid=canvas_grid,
        glimpse_px=cfg.glimpse_resolution_px,
    )
    policy_kwargs = {"probe": probe, "get_spatial_fn": seg.canvit.get_spatial} if policy == "entropy_coarse_to_fine" else {}

    log.info("  canvas_grid=%d  probe=%s  policy=%s  T=%d  glimpse_px=%d  batch=%d%s",
             canvas_grid, probe_repo, policy, T, cfg.glimpse_resolution_px, effective_bs,
             f" (was {cfg.batch_size})" if effective_bs != cfg.batch_size else "")

    inter_all   = torch.zeros(T, n_images, n_classes)
    union_all   = torch.zeros(T, n_images, n_classes)
    gt_area_all = torch.zeros(n_images, n_classes)  # GT area is timestep-invariant

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

            # Chunked upsample keeps peak memory bounded. Upsampling the full
            # batch to 512² at canvas_grid=64 × B=32 is ~5 GB; 8-at-a-time is
            # ~1.25 GB peak regardless of canvas_grid.
            if logits.shape[-1] != masks_dev.shape[-1]:
                preds = torch.empty((B, cfg.scene_size_px, cfg.scene_size_px),
                                     dtype=torch.int64, device=device)
                for c0 in range(0, B, _UPSAMPLE_CHUNK):
                    c1 = min(c0 + _UPSAMPLE_CHUNK, B)
                    up = F.interpolate(
                        logits[c0:c1], size=(cfg.scene_size_px, cfg.scene_size_px),
                        mode="bilinear", align_corners=False,
                    )
                    preds[c0:c1] = up.argmax(dim=1)
            else:
                preds = logits.argmax(dim=1)

            # One scatter_add for the whole batch (replaces B per-image histc calls).
            inter, union, gt_area = _batch_confusion(preds, masks_dev, n_classes)
            inter_all[step.t, img_start:img_start + B] = inter.cpu()
            union_all[step.t, img_start:img_start + B] = union.cpu()
            if step.t == 0:
                gt_area_all[img_start:img_start + B] = gt_area.cpu()
            del logits, preds

        img_start += B

    assert img_start == n_images, (img_start, n_images)
    _miou_sanity(inter_all[0], union_all[0], f"c{canvas_grid} t=0")
    _miou_sanity(inter_all[T - 1], union_all[T - 1], f"c{canvas_grid} t={T-1}")

    frames = []
    for t in range(T):
        df = _to_long_df(inter_all[t], union_all[t], gt_area_all, {
            "canvas_resolution": canvas_grid,
            "timestep": t,
        }, mask_resolution_px=cfg.scene_size_px, resize_mode=cfg.resize_mode)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    assert len(result) > 0, f"empty result for canvas_grid={canvas_grid}"
    assert result["timestep"].nunique() == T, (result["timestep"].nunique(), T)
    log.info("  canvas_grid=%d done in %.1fs  (%d rows across %d timesteps)",
             canvas_grid, time.monotonic() - t0, len(result), T)

    # policy_kwargs holds a bound method (seg.canvit.get_spatial) which keeps the entire
    # model graph alive; must be deleted explicitly before gc.collect() can free it.
    del seg, probe, policy_kwargs, loader, dataset, inter_all, union_all, gt_area_all, frames
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def _build_subprocess_cmd(cfg: CanViTConfig, canvas_resolution: int) -> list[str]:
    cmd = [sys.executable, "-m", "canvit_eval.tasks.ade20k_obj.iou", "canvit"]
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
    existing = pd.read_parquet(cfg.output) if cfg.output.exists() else pd.DataFrame()
    meta: dict[str, Any] = {
        "model_repo": cfg.model_repo,
        "canvas_resolutions_in_parquet": resolutions_in_parquet,
        "probe_repos": {cr: CANVIT_PROBE_REPOS[cr] for cr in resolutions_in_parquet if cr in CANVIT_PROBE_REPOS},
        "policies": {cr: _canvas_policy(cr) for cr in resolutions_in_parquet},
        "n_timesteps": cfg.n_timesteps,
        "scene_size_px": cfg.scene_size_px,
        "resize_mode": cfg.resize_mode,
        "glimpse_resolution_px": cfg.glimpse_resolution_px,
        "n_rows": int(len(existing)),
        "n_classes_with_data": int(existing["class_idx"].nunique()) if not existing.empty else 0,
        "timesteps_in_parquet": sorted(int(t) for t in existing["timestep"].unique()) if not existing.empty else [],
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    sidecar.write_text(json.dumps(meta, indent=2))
    log.info("sidecar → %s", sidecar)


def run_canvit(cfg: CanViTConfig) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    if not cfg.canvas_resolutions:
        raise ValueError(
            f"Specify --canvas-resolutions (any subset of {sorted(CANVIT_PROBE_REPOS)})"
        )
    unknown = [c for c in cfg.canvas_resolutions if c not in CANVIT_PROBE_REPOS]
    if unknown:
        raise ValueError(f"No probe repo for canvas_resolution={unknown}. Available: {sorted(CANVIT_PROBE_REPOS)}")
    assert cfg.n_timesteps > 0, cfg.n_timesteps

    # Multi-canvas: one subprocess per resolution so CUDA memory is fully released
    # between runs (in-process cleanup is best-effort; subprocess is bulletproof).
    if len(cfg.canvas_resolutions) > 1:
        log.info("=== CanViT per-(image, class, timestep) IoU ===")
        log.info("model_repo=%s", cfg.model_repo)
        log.info("canvas_resolutions=%s  n_timesteps=%d  glimpse_px=%d",
                 cfg.canvas_resolutions, cfg.n_timesteps, cfg.glimpse_resolution_px)
        log.info("ade20k_root=%s  scene_size_px=%d  resize_mode=%s",
                 cfg.ade20k_root_path, cfg.scene_size_px, cfg.resize_mode)
        log.info("output=%s", cfg.output)

        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        for i, cr in enumerate(cfg.canvas_resolutions):
            log.info("── subprocess %d/%d: canvas_resolution=%d ──",
                     i + 1, len(cfg.canvas_resolutions), cr)
            subprocess.run(_build_subprocess_cmd(cfg, cr), check=True)
        assert cfg.output.exists(), f"no output parquet produced: {cfg.output}"
        existing = pd.read_parquet(cfg.output)
        resolutions_in_parquet = sorted(int(c) for c in existing["canvas_resolution"].unique())
        missing = set(cfg.canvas_resolutions) - set(resolutions_in_parquet)
        assert not missing, f"after subprocesses, these canvas resolutions are missing from parquet: {sorted(missing)}"
        log.info("all %d subprocesses done in %.1fs  (parquet: %d rows, %d canvases)",
                 len(cfg.canvas_resolutions), time.monotonic() - t0,
                 len(existing), len(resolutions_in_parquet))
        _write_canvit_sidecar(cfg, resolutions_in_parquet)
        return cfg.output

    # Single canvas: run in-process.
    device = torch.device(cfg.device)
    (cr,) = cfg.canvas_resolutions
    log.info("=== CanViT per-(image, class, timestep) IoU ===")
    log.info("device=%s  canvas_resolution=%d  model_repo=%s", device, cr, cfg.model_repo)
    log.info("n_timesteps=%d  glimpse_px=%d  scene_size_px=%d  resize_mode=%s",
             cfg.n_timesteps, cfg.glimpse_resolution_px, cfg.scene_size_px, cfg.resize_mode)
    log.info("area_parquet=%s", cfg.area_parquet)
    log.info("output=%s", cfg.output)

    area_df = _load_area_df(cfg.area_parquet)
    assert len(area_df) > 0, f"area parquet is empty: {cfg.area_parquet}"
    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    if cfg.output.exists():
        existing = pd.read_parquet(cfg.output)
        kept = existing[existing["canvas_resolution"] != cr]
        log.info("existing parquet has %d rows across canvases %s; keeping %d rows (other canvases)",
                 len(existing), sorted(existing["canvas_resolution"].unique()), len(kept))
    else:
        kept = pd.DataFrame()

    df = _run_one_canvas(cr, cfg, device)
    df = df.merge(area_df, on=["image_idx", "class_idx"], how="left")
    assert not df["area"].isna().any(), "merge lost area for some rows"
    combined = pd.concat([kept, df], ignore_index=True) if not kept.empty else df
    combined.to_parquet(cfg.output, index=False)
    log.info("saved %s — parquet now %d rows across %d canvases (%.1f MB)",
             cfg.output, len(combined),
             combined["canvas_resolution"].nunique(),
             cfg.output.stat().st_size / 1e6)

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
