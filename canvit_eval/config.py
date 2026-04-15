"""Shared evaluation configuration — single source of truth for defaults.

Dataset paths: env var override > autodetect from known machine paths > error.
"""

import functools
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from canvit_eval.policies import PolicyName

log = logging.getLogger(__name__)

# Canonical pretrained CanViT-B repo — IN21K, additive canvas, VPE, DINOv3-B/16 teacher.
# Single source of truth; consumed by EpisodeConfig + all tasks that load the pretrained backbone.
DEFAULT_PRETRAINED_REPO = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"

# DINOv3 teacher repo.
TEACHER_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _resolve_path(env_var: str, known_paths: list[str], description: str) -> Path:
    """Resolve a dataset path: env var > known paths > error. Cached + logged."""
    v = os.environ.get(env_var)
    if v is not None:
        log.info("%s from env: %s", description, v)
        return Path(v)
    for p in known_paths:
        if Path(p).is_dir():
            log.info("%s autodetected: %s", description, p)
            return Path(p)
    raise RuntimeError(
        f"{description} not found. Set {env_var} env var, "
        f"or ensure one of {known_paths} exists."
    )


@functools.cache
def ade20k_root() -> Path:
    return _resolve_path(
        env_var="ADE20K_ROOT",
        known_paths=["/datasets/ADE20k/ADEChallengeData2016"],  # crockett
        description="ADE20K root",
    )


@functools.cache
def imagenet_val_dir() -> Path:
    return _resolve_path(
        env_var="IMAGENET_VAL",
        known_paths=[
            "/datasets/ILSVRC/Data/CLS-LOC/val",       # crockett
            "/datashare/imagenet/ILSVRC2012/val",       # nibi
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
    glimpse_px: int = 128
    min_scale: float = 0.05
    max_scale: float = 1.0
