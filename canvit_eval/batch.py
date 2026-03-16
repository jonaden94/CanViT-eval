"""Batch evaluation: reproduce ALL paper ADE20K results.

The eval matrix is derived from a minimal config. No duplication.

Usage:
    ADE20K_ROOT=... uv run python -m canvit_eval.batch
    ADE20K_ROOT=... uv run python -m canvit_eval.batch --n-runs 1  # quick
"""

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

log = logging.getLogger(__name__)

POLICIES = ["coarse_to_fine", "fine_to_coarse", "full_then_random",
            "random", "entropy_coarse_to_fine", "constant_full_scene"]
DETERMINISTIC = {"constant_full_scene"}
RESOLUTIONS = [(512, 32, 32), (1024, 64, 8)]  # (scene, grid, batch_size)
DINOV3_VARIANTS = ["dv3b", "dv3s"]
DINOV3_RESOLUTIONS = [128, 144, 160, 192, 256, 384, 512]
CANVAS_GRIDS = [(512, 8), (512, 16), (512, 32), (1024, 64)]


def _probe(scene: int, grid: int) -> str:
    return f"canvit/probe-ade20k-40k-s{scene}-c{grid}-in21k"


def _run(args: list[str], out: Path) -> None:
    if out.exists():
        log.info("SKIP %s", out.name)
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

    # CanViT multi-timestep policy evals
    for scene, grid, bs in RESOLUTIONS:
        for policy in POLICIES:
            for run in range(args.n_runs):
                if policy in DETERMINISTIC and run > 0:
                    continue
                out = d / f"{policy}_s{scene}_c{grid}_run{run}.pt"
                _run(["ade20k-seg", "--probe-repo", _probe(scene, grid),
                      "--episode.policy", policy, "--episode.n-timesteps", str(args.n_timesteps),
                      "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                      "--batch-size", str(bs), "--output", str(out)], out)

    # DINOv3 baseline probes
    for v in DINOV3_VARIANTS:
        for res in DINOV3_RESOLUTIONS:
            out = d / f"{v}_{res}px.pt"
            _run(["ade20k-seg", "--model", "dinov3",
                  "--probe-repo", f"canvit/probe-ade20k-40k-{v}-{res}px",
                  "--eval-resolution", str(res), "--output", str(out)], out)

    # CanViT single-glimpse probes (beating teacher table)
    for scene, grid in CANVAS_GRIDS:
        out = d / f"canvit_s{scene}-c{grid}-in21k.pt"
        _run(["ade20k-seg", "--probe-repo", _probe(scene, grid),
              "--episode.policy", "coarse_to_fine", "--episode.n-timesteps", "1",
              "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
              "--output", str(out)], out)

    log.info("DONE")


if __name__ == "__main__":
    main(tyro.cli(Args))
