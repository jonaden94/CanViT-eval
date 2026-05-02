# CanViT-eval

Evaluation and benchmarking for CanViT. Produces `.pt` result files consumed by
the paper-export pipeline for figures and tables. All models and probes can load
from HuggingFace Hub or from a local checkpoint mirror.

Dataset paths autodetect from a small set of common local mounts; override with
`ADE20K_ROOT` / `IMAGENET_VAL` env vars otherwise (see `.envrc.example`).

## Hardware notes

Tested on RTX 4090 (24 GB) and H100 SXM (80 GB). Default per-config batch sizes
in `canvit_eval/batch.py::_BATCH_SIZE_BY_SCENE` assume ≥16 GB VRAM. On smaller
GPUs (e.g. 8 GB), expect OOM on `in1k-clf` (default batch 64 at s=512/c=32) and
on multi-step `ade20k-seg` policy curves; override the per-config batch size in
that table or invoke single tasks with a smaller `--batch-size`.

## Usage

### Single eval

Each subcommand is a task Config dataclass with a `run()` method; tyro
generates the CLI.

```bash
# CanViT ADE20K segmentation (multi-timestep episode):
uv run python -m canvit_eval ade20k-seg-canvit \
    --probe-repo canvit/probe-ade20k-40k-s512-c32-in21k \
    --episode.policy coarse_to_fine --episode.n-timesteps 21 \
    --output results/ade20k_seg/my_run.pt

# DINOv3 ADE20K passive baseline (single forward pass):
uv run python -m canvit_eval ade20k-seg-dinov3 \
    --probe-repo canvit/probe-ade20k-40k-dv3b-256px \
    --eval-resolution 256 \
    --output results/ade20k_seg/dv3b_256px.pt

# ImageNet-1k classification:
uv run python -m canvit_eval in1k-clf --mode finetuned

# Reconstruction quality (ablation checkpoints):
uv run python -m canvit_eval reconstruction \
    --model-repo canvit/canvitb16-abl-baseline-2026-03-02
```

### Batch eval matrix

```bash
# Run the eval matrix with N runs per stochastic policy (sequential, single GPU).
# Deterministic policies are auto-trimmed to n=1; stochastic policies draw from
# the default RNG (no explicit seed setting).
uv run python -m canvit_eval.batch --n-runs N

# Filter:
uv run python -m canvit_eval.batch --tasks ade20k-seg --grids 32 --policies coarse_to_fine

# Include the newly-trained c9/10/12/24 @ s512 probes:
uv run python -m canvit_eval.batch --include-extra-grids

# Preview commands without running:
uv run python -m canvit_eval.batch --dry-run
```

### Incremental pooling with `--skip-existing`

`--skip-existing` matches prior outputs by structural identity (task, model,
policy, scene, grid, run_idx) via a timestamp-agnostic glob, so crashed
batches resume cleanly AND run counts pool over time:

```bash
# Day 1: preliminary n=1 on a new grid.
uv run python -m canvit_eval.batch --tasks ade20k-seg --grids 9 --include-extra-grids --n-runs 1

# Day 2: add 4 more runs without rerunning r=0. Export bootstraps over all 5.
uv run python -m canvit_eval.batch --tasks ade20k-seg --grids 9 --include-extra-grids --n-runs 5 --skip-existing
```

Each run writes `{stem}_{UTC_ts}_r{run}.pt`. The paper's export preserves
all runs (even same run_idx across different timestamps) and logs per-file
provenance (timestamp, git_commit, model_repo, final-t mIoU) — see
`export/ade20k_seg.py:_build_policy_curves` for the audit line format.

## Latency bench

Per-forward-pass latency measurements for the paper's hardware table.
Separate from the eval batch (no probes, no datasets — just model forward
timing). Full doc: `bench/pt/README.md`.

```bash
# Fast profile (~15 min, paper figure).
uv run python bench/pt/matrix.py --profile fast
# Full profile (~25 min, 3 passes for distributional analysis).
uv run python bench/pt/matrix.py --profile full

# Analysis over the resulting JSONLs:
uv run python bench/pt/analyze.py --pattern 'bench/pt/results/*.jsonl'
```

## Architecture

```bash
uv run pypatree
```

## Tests

```bash
uv run pytest
```

## Related repos

| Repo | Role |
|------|------|
| [CanViT-PyTorch](https://github.com/m2b3/CanViT-PyTorch) (public, canonical) | Core model (`canvit_pytorch` package), probe architecture (`canvit_pytorch.probes`) |
| [CanViT-specialize](https://github.com/m2b3/CanViT-specialize) | Probe training, ADE20K dataloader, IoU metric |
| [CanViT-pretrain](https://github.com/m2b3/CanViT-pretrain) | Model pretraining |
| Paper repository | Paper (.pt → JSON → PDF) |
