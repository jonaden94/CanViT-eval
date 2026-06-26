"""Shared evaluation configuration.

Dataset paths resolve as env var override > known machine paths > generic
fallback. Real runs validate the resolved paths where the data is opened so
CLI help remains usable on machines without the datasets mounted.
"""

import functools
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from canvit_pytorch import resolve_canvit_repo

from canvit_eval.policies import PolicyName

log = logging.getLogger(__name__)

# Pretrained CanViT-B repo (IN21k, additive canvas, VPE, DINOv3-B/16 teacher).
DEFAULT_PRETRAINED_REPO = resolve_canvit_repo("canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02")

# DINOv3 teacher repos (public, third-party — no resolve wrap).
DINOV3_VITB_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_VITS_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"
TEACHER_REPO = DINOV3_VITB_REPO


def _resolve_path(env_var: str, known_paths: list[str], description: str) -> Path:
    """Resolve a dataset path without touching the filesystem on fallback."""
    v = os.environ.get(env_var)
    if v is not None:
        log.info("%s from env: %s", description, v)
        return Path(v)
    for p in known_paths:
        if Path(p).is_dir():
            log.info("%s autodetected: %s", description, p)
            return Path(p)
    fallback = Path(known_paths[0])
    log.info("%s defaulting to %s; validate before use", description, fallback)
    return fallback


def require_existing_dir(path: Path, *, description: str, env_var: str | None = None) -> None:
    if path.is_dir():
        return
    hint = f" Set {env_var} or pass the corresponding CLI path." if env_var else ""
    raise RuntimeError(f"{description} not found at {path}.{hint}")


@functools.cache
def ade20k_root() -> Path:
    return _resolve_path(
        env_var="ADE20K_ROOT",
        known_paths=["/datasets/ADE20k/ADEChallengeData2016"],
        description="ADE20K root",
    )


@functools.cache
def imagenet_val_dir() -> Path:
    return _resolve_path(
        env_var="IMAGENET_VAL",
        known_paths=[
            "/datasets/ILSVRC/Data/CLS-LOC/val",
            "/datashare/imagenet/ILSVRC2012/val",
        ],
        description="ImageNet val dir",
    )


@dataclass(frozen=True)
class EpisodeConfig:
    """How to run CanViT episodes. Shared by all CanViT-based tasks."""

    model_repo: str = DEFAULT_PRETRAINED_REPO
    policy: PolicyName = "coarse_to_fine"
    n_timesteps: int = 21
    canvas_grid: int | None = None  # None → scene_size // patch_size
    # Pixel size of the glimpse crop fed to the uniform patcher. ``None`` (default)
    # derives it from the model as ``glimpse_grid_size × patch_size_px`` — the SAME
    # value training used (train/model.py) — so it auto-tracks the backbone's patch
    # size (8px → 64, 16px → 128, 6px → 48, …). Set an int only to force a specific
    # crop size. Ignored by foveated/square patchers (they consume the full image).
    glimpse_px: int | None = None
    min_scale: float = 0.05
    max_scale: float = 1.0
    # Eval-time view-scale override, patcher- and policy-agnostic. ``None``
    # (default) passes the policy's own scales through (e.g. coarse-to-fine
    # actually zooms full → 0.5 → 0.25). A float pins EVERY glimpse to that
    # scale while keeping the policy's centers — for the uniform patcher this
    # fixes the pre-crop zoom; for the foveated/square patchers (which honor
    # viewpoint.scales, fix_size = scale * H) it fixes the sensor window
    # (1.0 = full image, 0.8 / 1.41 / … = a fixed zoom). Use a fixed value to
    # evaluate a fixed-scale or per-rollout model in-distribution, and ``None``
    # to let a per-glimpse (multi-scale) model follow the policy's zoom.
    override_scale: float | None = None
