"""Batch evaluation: assemble and run the eval matrix for the paper.

Results go to task-specific subdirs under --out-dir (default: results/):
    ade20k_seg/
    in1k_clf_{frozen,finetuned}/
    recon/

Filenames include a UTC timestamp for provenance; skip-existing matches on the
structural identity (task, model, policy, scene, grid, run_idx), not on
filename equality — so reruns can resume cleanly without caring about
when the prior run happened.

Usage:

    # Full paper matrix, 5 seeds, single GPU, sequential:
    uv run python -m canvit_eval.batch --n-runs 5

    # Resume: skip any job whose structural output already exists on disk:
    uv run python -m canvit_eval.batch --n-runs 5 --skip-existing

    # Filter by task / grid / policy:
    uv run python -m canvit_eval.batch --tasks ade20k-seg --grids 32
    uv run python -m canvit_eval.batch --policies coarse_to_fine entropy_coarse_to_fine

    # Extend the ADE20K matrix with freshly trained canvas grids (c9/10/12/24 @ s512):
    uv run python -m canvit_eval.batch --include-extra-grids

    # Preview without running:
    uv run python -m canvit_eval.batch --dry-run

    # SLURM array (one cluster job per row) — not used currently, Nibi allocation
    # is no longer available, crockett is the primary eval machine:
    uv run python -m canvit_eval.batch --dry-run > /tmp/eval_cmds.txt
    sbatch --array=1-$(wc -l < /tmp/eval_cmds.txt) slurm/array.sbatch
"""

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import tyro

from canvit_eval.policies import IN1K_POLICIES, PolicyName

log = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


# ── ADE20K segmentation matrix ─────────────────────────────────────────

ALL_POLICIES: list[PolicyName] = list(get_args(PolicyName))

# Policies whose output is bit-identical across runs, given a fixed probe + fixed
# dataloader order. The batch builder trims n_runs → 1 for these to avoid waste.
#
#   repeated_full_scene     — static viewpoint sequence, no RNG.
#   entropy_coarse_to_fine  — argmax over entropy scores (policies.py:140),
#                             no RNG introduced anywhere in the episode.
DETERMINISTIC: frozenset[PolicyName] = frozenset({"repeated_full_scene", "entropy_coarse_to_fine"})


def _n_runs_for(policy: PolicyName, n_runs: int) -> int:
    return 1 if policy in DETERMINISTIC else n_runs


# (scene_size, canvas_grid, batch_size) for CanViT policy-curve evals.
# Paper's canonical matrix. See EXTRA_ADE20K_RESOLUTIONS for the opt-in extension.
ADE20K_RESOLUTIONS: list[tuple[int, int, int]] = [
    (512, 32, 32),
    (1024, 64, 8),
]

# (scene_size, canvas_grid) for CanViT single-glimpse (t=0) probes — the passive-comparison table.
# batch_size lookup is shared with ADE20K_RESOLUTIONS via _BATCH_SIZE_BY_SCENE.
CANVAS_GRIDS: list[tuple[int, int]] = [
    (512, 8), (512, 16), (512, 32), (1024, 64),
]

# Opt-in via --include-extra-grids. These are the c9/10/12/24 @ s512 probes trained 2026-04-14
# to match the DINOv3 baseline resolutions {144, 160, 192, 384} px. Not in default runs so
# arXiv-v1 reproduction remains identical by default.
EXTRA_CANVAS_GRIDS: list[tuple[int, int]] = [
    (512, 9), (512, 10), (512, 12), (512, 24),
]
EXTRA_ADE20K_RESOLUTIONS: list[tuple[int, int, int]] = [(s, g, 32) for s, g in EXTRA_CANVAS_GRIDS]

# DINOv3 passive baselines: 2 variants × 7 resolutions = 14 evals.
# 768/1024 px DINOv3 probes would be required to match CanViT c48@s512 and c64@s1024 data
# points; not currently trained (would need a different training machine).
DINOV3_VARIANTS: dict[str, str] = {
    "dv3b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dv3s": "facebook/dinov3-vits16-pretrain-lvd1689m",
}
DINOV3_RESOLUTIONS: list[int] = [128, 144, 160, 192, 256, 384, 512]

# Single source of truth: derive {scene → batch_size} from the combined matrix.
# Hard-crashes on unknown scene (no silent fallback — missing entry = code bug).
_BATCH_SIZE_BY_SCENE: dict[int, int] = {
    scene: bs for scene, _, bs in ADE20K_RESOLUTIONS + EXTRA_ADE20K_RESOLUTIONS
}


# ── Ablation reconstruction matrix ─────────────────────────────────────
# Single source of truth for ablation HF repo IDs; the paper repo imports this
# via `from canvit_eval.batch import ABLATION_REPOS` for FLOP/param analysis.

ABLATION_REPOS: dict[str, str] = {
    "baseline":      "canvit/canvitb16-abl-baseline-2026-03-02",
    "qkvo-dcan256":  "canvit/canvitb16-abl-qkvo-dcan256-2026-03-02",
    "qkvo-dcan384":  "canvit/canvitb16-abl-qkvo-dcan384-2026-03-02",
    "dcan256":       "canvit/canvitb16-abl-dcan256-2026-03-02",
    "no-dense":      "canvit/canvitb16-abl-no-dense-2026-03-02",
    "no-fiid-1riid": "canvit/canvitb16-abl-no-fiid-1riid-2026-03-02",
    "no-fiid-2riid": "canvit/canvitb16-abl-no-fiid-2riid-2026-03-06",
    "no-bptt":       "canvit/canvitb16-abl-no-bptt-2026-03-06",
    "no-reads":      "canvit/canvitb16-abl-no-reads-2026-03-02",
    "no-vpe":        "canvit/canvitb16-abl-no-vpe-2026-03-03",
    "rw-stride6":    "canvit/canvitb16-abl-rw-stride6-2026-03-03",
    "vit-s":         "canvit/canvitb16-abl-vit-s-2026-03-03",
}


def _probe_repo(scene: int, grid: int) -> str:
    return f"canvit/probe-ade20k-40k-s{scene}-c{grid}-in21k"


# ── Job model ──────────────────────────────────────────────────────────

TaskName = Literal["ade20k-seg", "in1k-clf", "recon"]
ALL_TASKS: list[TaskName] = list(get_args(TaskName))


@dataclass(frozen=True)
class EvalJob:
    """One evaluation invocation.

    Structural fields (task/model/policy/scene_size/canvas_grid/input_px/run_idx)
    drive filter_jobs(), skip_existing matching, and per-job logging — so we
    never parse filenames at runtime. `args` is what's passed to canvit_eval CLI.
    `output_stem` is the timestamp-free structural identity used to check for
    prior outputs (any .pt under output_dir/output_stem*.pt counts as done).
    """
    task: TaskName
    args: list[str]
    output: Path
    output_stem: str          # filename without timestamp, glob-matched for skip-existing
    model: str                # "canvit" | "canvit-finetuned" | "dinov3-b" | "dinov3-s" | ablation-slug
    policy: PolicyName | None = None
    scene_size: int | None = None
    canvas_grid: int | None = None
    input_px: int | None = None
    run_idx: int = 0

    def already_done(self) -> bool:
        """True iff any .pt matching this job's structural identity exists
        in the output dir (timestamp-agnostic glob match)."""
        return any(self.output.parent.glob(f"{self.output_stem}*.pt"))

    def describe(self) -> str:
        parts = [self.task, f"model={self.model}"]
        if self.policy:      parts.append(f"policy={self.policy}")
        if self.scene_size:  parts.append(f"s={self.scene_size}")
        if self.canvas_grid: parts.append(f"c={self.canvas_grid}")
        if self.input_px:    parts.append(f"input={self.input_px}px")
        parts.append(f"r={self.run_idx}")
        return " ".join(parts)


# ── Matrix builders ────────────────────────────────────────────────────


def _ade20k_seg_jobs(
    out_dir: Path, *, n_runs: int, n_timesteps: int, ts: str,
    ade20k_res: list[tuple[int, int, int]], canvas_grids: list[tuple[int, int]],
) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    d = out_dir / "ade20k_seg"

    # (a) CanViT multi-timestep policy evals.
    for scene, grid, bs in ade20k_res:
        probe = _probe_repo(scene, grid)
        for policy in ALL_POLICIES:
            for run in range(_n_runs_for(policy, n_runs)):
                stem = f"{policy}_s{scene}_c{grid}"
                out = d / f"{stem}_{ts}_r{run}.pt"
                jobs.append(EvalJob(
                    task="ade20k-seg",
                    args=["ade20k-seg-canvit", "--probe-repo", probe,
                          "--episode.policy", policy, "--episode.n-timesteps", str(n_timesteps),
                          "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                          "--batch-size", str(bs), "--output", str(out)],
                    output=out, output_stem=f"{stem}_",
                    model="canvit", policy=policy,
                    scene_size=scene, canvas_grid=grid, run_idx=run,
                ))

    # (b) DINOv3 baselines (single passive forward, deterministic → n_runs=1).
    for variant, teacher_repo in DINOV3_VARIANTS.items():
        for res in DINOV3_RESOLUTIONS:
            stem = f"{variant}_{res}px"
            out = d / f"{stem}_{ts}.pt"
            jobs.append(EvalJob(
                task="ade20k-seg",
                args=["ade20k-seg-dinov3",
                      "--probe-repo", f"canvit/probe-ade20k-40k-{variant}-{res}px",
                      "--teacher-repo", teacher_repo,
                      "--eval-resolution", str(res), "--output", str(out)],
                output=out, output_stem=f"{stem}_",
                model=f"dinov3-{variant[-1]}",
                input_px=res, canvas_grid=res // 16,
            ))

    # (c) CanViT single-glimpse probes (t=0, full-scene viewpoint — deterministic).
    for scene, grid in canvas_grids:
        bs = _BATCH_SIZE_BY_SCENE[scene]
        stem = f"canvit_s{scene}_c{grid}"
        out = d / f"{stem}_{ts}.pt"
        jobs.append(EvalJob(
            task="ade20k-seg",
            args=["ade20k-seg-canvit", "--probe-repo", _probe_repo(scene, grid),
                  "--episode.policy", "coarse_to_fine", "--episode.n-timesteps", "1",
                  "--episode.canvas-grid", str(grid), "--scene-size", str(scene),
                  "--batch-size", str(bs), "--output", str(out)],
            output=out, output_stem=f"{stem}_",
            model="canvit", policy="coarse_to_fine",
            scene_size=scene, canvas_grid=grid, input_px=128,
        ))

    return jobs


def _in1k_clf_jobs(out_dir: Path, *, n_runs: int, n_timesteps: int, ts: str, mode: Literal["frozen", "finetuned"]) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    d = out_dir / f"in1k_clf_{mode}"
    for policy in IN1K_POLICIES:
        for run in range(_n_runs_for(policy, n_runs)):
            stem = f"in1k_{policy}"
            out = d / f"{stem}_{ts}_r{run}.pt"
            jobs.append(EvalJob(
                task="in1k-clf",
                args=["in1k-clf", "--mode", mode,
                      "--episode.policy", policy, "--episode.n-timesteps", str(n_timesteps),
                      "--output", str(out)],
                output=out, output_stem=f"{stem}_",
                model=f"canvit-{mode}", policy=policy, run_idx=run,
            ))
    return jobs


def _recon_jobs(out_dir: Path, *, n_runs: int, ts: str) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    d = out_dir / "recon"
    for slug, repo in ABLATION_REPOS.items():
        for run in range(n_runs):
            stem = f"recon_{slug}"
            out = d / f"{stem}_{ts}_r{run}.pt"
            jobs.append(EvalJob(
                task="recon",
                args=["reconstruction", "--model-repo", repo, "--output", str(out)],
                output=out, output_stem=f"{stem}_",
                model=slug, run_idx=run,
            ))
    return jobs


def build_eval_matrix(
    out_dir: Path, *,
    n_runs: int,
    n_timesteps: int,
    tasks: list[TaskName],
    include_extra_grids: bool = False,
) -> list[EvalJob]:
    """Generate all eval jobs. Pure — no side effects except the timestamp."""
    ts = _utc_timestamp()
    ade20k_res = ADE20K_RESOLUTIONS + (EXTRA_ADE20K_RESOLUTIONS if include_extra_grids else [])
    canvas_grids = CANVAS_GRIDS + (EXTRA_CANVAS_GRIDS if include_extra_grids else [])

    jobs: list[EvalJob] = []
    if "ade20k-seg" in tasks:
        jobs.extend(_ade20k_seg_jobs(
            out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts,
            ade20k_res=ade20k_res, canvas_grids=canvas_grids,
        ))
    if "in1k-clf" in tasks:
        jobs.extend(_in1k_clf_jobs(out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts, mode="frozen"))
        jobs.extend(_in1k_clf_jobs(out_dir, n_runs=n_runs, n_timesteps=n_timesteps, ts=ts, mode="finetuned"))
    if "recon" in tasks:
        jobs.extend(_recon_jobs(out_dir, n_runs=n_runs, ts=ts))
    return jobs


def filter_jobs(
    jobs: list[EvalJob], *,
    policies: list[str] | None = None,
    grids: list[int] | None = None,
) -> list[EvalJob]:
    """Keep jobs matching every supplied filter. Each filter drops jobs whose
    corresponding field is None (i.e., a grid filter drops DINOv3 jobs with
    `canvas_grid=res//16` still counts — those do have a canvas_grid). None/[]
    filter = pass-through for that dimension."""
    out = jobs
    if policies:
        policy_set = set(policies)
        out = [j for j in out if j.policy is not None and j.policy in policy_set]
    if grids:
        grid_set = set(grids)
        out = [j for j in out if j.canvas_grid is not None and j.canvas_grid in grid_set]
    return out


# ── CLI ────────────────────────────────────────────────────────────────


@dataclass
class Args:
    out_dir: Path = Path("results")
    n_runs: int = 5
    n_timesteps: int = 21
    tasks: list[TaskName] = field(default_factory=lambda: list(ALL_TASKS))
    dry_run: bool = False
    skip_existing: bool = False
    """Skip jobs whose structural output already exists (timestamp-agnostic glob match)."""
    include_extra_grids: bool = False
    """Include c9/10/12/24 @ s512 probes in the ADE20K matrix (default: paper-v1 set only)."""
    policies: list[str] = field(default_factory=list)
    """Filter to these policies (empty = all). Applies to jobs that carry a policy field."""
    grids: list[int] = field(default_factory=list)
    """Filter to these canvas grids (empty = all). For DINOv3: grid = input_px // 16."""


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    jobs = build_eval_matrix(
        args.out_dir,
        n_runs=args.n_runs, n_timesteps=args.n_timesteps, tasks=args.tasks,
        include_extra_grids=args.include_extra_grids,
    )
    jobs = filter_jobs(jobs, policies=args.policies or None, grids=args.grids or None)

    n_total = len(jobs)
    if args.skip_existing:
        jobs = [j for j in jobs if not j.already_done()]
    n_skipped = n_total - len(jobs)

    by_task: dict[str, int] = {}
    for j in jobs:
        by_task[j.task] = by_task.get(j.task, 0) + 1
    log.info("%d jobs to run (%d skipped as already done)%s",
             len(jobs), n_skipped,
             ": " + ", ".join(f"{k}={v}" for k, v in sorted(by_task.items())) if by_task else "")

    if args.dry_run:
        for job in jobs:
            print(" ".join(["uv", "run", "python", "-m", "canvit_eval"] + job.args))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    total_elapsed = 0.0
    for i, job in enumerate(jobs, start=1):
        t0 = time.monotonic()
        log.info("[%d/%d] RUN  %s", i, len(jobs), job.describe())
        result = subprocess.run([sys.executable, "-m", "canvit_eval"] + job.args)
        elapsed = time.monotonic() - t0
        total_elapsed += elapsed
        if result.returncode != 0:
            log.error("[%d/%d] FAIL (%.1fs, exit %d)  %s", i, len(jobs), elapsed, result.returncode, job.output.name)
            failed += 1
        else:
            log.info("[%d/%d] OK   (%.1fs)  %s", i, len(jobs), elapsed, job.output.name)
            done += 1

    log.info("DONE: %d/%d completed, %d failed, %.1f min total",
             done, len(jobs), failed, total_elapsed / 60)


if __name__ == "__main__":
    main(tyro.cli(Args))
