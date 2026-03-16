"""Shared evaluation configuration — single source of truth for defaults.

No machine-specific defaults. Paths must be provided explicitly or via env vars.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from canvit_eval.policies import PolicyName

# Canonical model repo — used as default everywhere.
DEFAULT_MODEL_REPO = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"

# DINOv3 teacher repo.
TEACHER_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def ade20k_root() -> Path:
    """ADE20K root from ADE20K_ROOT env var. Fails clearly if unset."""
    v = os.environ.get("ADE20K_ROOT")
    if v is None:
        raise RuntimeError("ADE20K_ROOT env var not set. Example: ADE20K_ROOT=/datasets/ADE20k/ADEChallengeData2016")
    return Path(v)


@dataclass(frozen=True)
class EpisodeConfig:
    """How to run CanViT episodes. Shared by all CanViT-based tasks."""

    model_repo: str = DEFAULT_MODEL_REPO
    policy: PolicyName = "coarse_to_fine"
    n_timesteps: int = 21
    canvas_grid: int | None = None  # None → scene_size // patch_size
    glimpse_px: int = 128
    min_scale: float = 0.05
    max_scale: float = 1.0
