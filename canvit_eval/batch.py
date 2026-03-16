"""Batch evaluation: reproduce ALL paper ADE20K results.

Runs all policy × resolution × run combinations via subprocess.
Skips existing output files (safe to interrupt and resume).

Usage:
    ADE20K_ROOT=... uv run python -m canvit_eval.batch
    ADE20K_ROOT=... uv run python -m canvit_eval.batch --n-runs 1  # quick smoke test
"""

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import tyro

from canvit_eval.policies import PolicyName

log = logging.getLogger(__name__)

# All policies from the PolicyName Literal — single source of truth.
ALL_POLICIES: list[str] = list(get_args(PolicyName))

# Policies with no randomness (only need 1 run).
DETERMINISTIC: set[str] = {"constant_full_scene"}

# (scene_px, canvas_grid, batch_size) — batch_size varies by VRAM needs.
RESOLUTIONS = [(512, 32, 32), (1024, 64, 8)]

# DINOv3 baseline probes.
DINOV3_VARIANTS = ["dv3b", "dv3s"]
DINOV3_RESOLUTIONS = [128, 144, 160, 192, 256, 384, 512]

# CanViT single-glimpse probes (beating teacher table).
CANVAS_GRIDS = [(512, 8), (512, 16), (512, 32), (1024, 64)]


def _probe_repo(scene: int, grid: int) -> str:
    return f"canvit/probe-ade20k-40k-s{scene}-c{grid}-in21k"


def _run(args: list[str], out: Path) -> None:
    if out.exists():
        log.info("SKIP %s (exists)", out.name)
        return
    log.info("RUN  %s", out.name)
    subprocess.run([sys.executable, "-m", "canvit_eval"] + args, check=True)


@dataclass
class Args:
    out_dir: Path = Path("results")
    n_runs: int = 5
    n_timesteps: int = 21


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    d = args.out_dir
    d.mkdir(parents=True, exist_ok=True)

    total = 0

    # CanViT multi-timestep policy evals
    for scene, grid, bs in RESOLUTIONS:
        probe = _probe_repo(scene, grid)
        for policy in ALL_POLICIES:
            n = 1 if policy in DETERMINISTIC else args.n_runs
            for run in range(n):
                out = d / f"{policy}_s{scene}_c{grid}_run{run}.pt"
                _run(["ade20k-seg", "--probe-repo", probe,
                      "--episode.policy", policy, "--episode.n-timesteps", str(args.n_timesteps),
                      "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                      "--batch-size", str(bs), "--output", str(out)], out)
                total += 1

    # DINOv3 baseline probes
    for v in DINOV3_VARIANTS:
        for res in DINOV3_RESOLUTIONS:
            out = d / f"{v}_{res}px.pt"
            _run(["ade20k-seg", "--model", "dinov3",
                  "--probe-repo", f"canvit/probe-ade20k-40k-{v}-{res}px",
                  "--eval-resolution", str(res), "--output", str(out)], out)
            total += 1

    # CanViT single-glimpse probes (beating teacher table)
    for scene, grid in CANVAS_GRIDS:
        out = d / f"canvit_s{scene}-c{grid}-in21k.pt"
        _run(["ade20k-seg", "--probe-repo", _probe_repo(scene, grid),
              "--episode.policy", "coarse_to_fine", "--episode.n-timesteps", "1",
              "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
              "--output", str(out)], out)
        total += 1

    log.info("DONE (%d evals scheduled)", total)


if __name__ == "__main__":
    main(tyro.cli(Args))
