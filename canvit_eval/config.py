"""Shared evaluation configuration — single source of truth for defaults.

All task configs compose with these fields. No duplication of default values.
"""

from dataclasses import dataclass, field
from pathlib import Path

from canvit_eval.policies import PolicyName

# Canonical model repo — used as default everywhere.
DEFAULT_MODEL_REPO = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"


@dataclass
class EpisodeConfig:
    """How to run CanViT episodes. Shared by all CanViT-based tasks."""

    model_repo: str = DEFAULT_MODEL_REPO
    policy: PolicyName = "coarse_to_fine"
    n_timesteps: int = 21
    canvas_grid: int | None = None  # None → scene_size // patch_size
    glimpse_px: int = 128
    min_scale: float = 0.05
    max_scale: float = 1.0


@dataclass
class HardwareConfig:
    """Hardware and performance settings. Shared by all tasks."""

    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True
