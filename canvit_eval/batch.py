"""Batch evaluation: reproduce ALL paper ADE20K results.

Generates the full eval matrix, then either runs sequentially (default)
or prints sbatch commands for parallel SLURM submission (--dry-run).

Usage:
    # Sequential (crockett, single GPU):
    ADE20K_ROOT=... uv run python -m canvit_eval.batch --n-runs 5

    # Print SLURM commands (Nibi, parallel):
    ADE20K_ROOT=... uv run python -m canvit_eval.batch --n-runs 5 --dry-run

    # Quick smoke test:
    ADE20K_ROOT=... uv run python -m canvit_eval.batch --n-runs 1
"""

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import tyro

from canvit_eval.policies import PolicyName

log = logging.getLogger(__name__)

def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

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


@dataclass
class EvalJob:
    """One evaluation to run."""
    args: list[str]
    output: Path


def build_eval_matrix(out_dir: Path, n_runs: int, n_timesteps: int) -> list[EvalJob]:
    """Generate the full list of eval jobs. Pure function, no side effects."""
    jobs: list[EvalJob] = []

    # CanViT multi-timestep policy evals
    for scene, grid, bs in RESOLUTIONS:
        probe = _probe_repo(scene, grid)
        for policy in ALL_POLICIES:
            n = 1 if policy in DETERMINISTIC else n_runs
            for run in range(n):
                out = out_dir / f"{policy}_s{scene}_c{grid}_run{run}.pt"
                jobs.append(EvalJob(
                    args=["ade20k-seg", "--probe-repo", probe,
                          "--episode.policy", policy, "--episode.n-timesteps", str(n_timesteps),
                          "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                          "--batch-size", str(bs), "--output", str(out)],
                    output=out,
                ))

    # DINOv3 baseline probes
    for v in DINOV3_VARIANTS:
        for res in DINOV3_RESOLUTIONS:
            out = out_dir / f"{v}_{res}px.pt"
            jobs.append(EvalJob(
                args=["ade20k-seg", "--model", "dinov3",
                      "--probe-repo", f"canvit/probe-ade20k-40k-{v}-{res}px",
                      "--eval-resolution", str(res), "--output", str(out)],
                output=out,
            ))

    # CanViT single-glimpse probes (beating teacher table)
    for scene, grid in CANVAS_GRIDS:
        out = out_dir / f"canvit_s{scene}-c{grid}-in21k.pt"
        jobs.append(EvalJob(
            args=["ade20k-seg", "--probe-repo", _probe_repo(scene, grid),
                  "--episode.policy", "coarse_to_fine", "--episode.n-timesteps", "1",
                  "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                  "--output", str(out)],
            output=out,
        ))

    return jobs


@dataclass
class Args:
    out_dir: Path = Path("results")
    n_runs: int = 5
    n_timesteps: int = 21
    dry_run: bool = False
    no_timestamp_dir: bool = False
    """If False (default), creates a timestamped subdirectory inside out_dir."""


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = args.out_dir
    if not args.no_timestamp_dir:
        out_dir = args.out_dir / _timestamp()
        log.info("Output directory: %s", out_dir)
    jobs = build_eval_matrix(out_dir, args.n_runs, args.n_timesteps)
    log.info("%d total eval jobs", len(jobs))

    if args.dry_run:
        for job in jobs:
            cmd = " ".join(["uv", "run", "python", "-m", "canvit_eval"] + job.args)
            print(cmd)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done, skipped = 0, 0
    for job in jobs:
        if job.output.exists():
            log.info("SKIP %s (exists)", job.output.name)
            skipped += 1
            continue
        log.info("RUN  %s", job.output.name)
        subprocess.run([sys.executable, "-m", "canvit_eval"] + job.args, check=True)
        done += 1

    log.info("DONE: %d run, %d skipped, %d total", done, skipped, len(jobs))


if __name__ == "__main__":
    main(tyro.cli(Args))
