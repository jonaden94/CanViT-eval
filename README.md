# CanViT-eval

Evaluation and benchmarking for CanViT. Produces `.pt` result files consumed by
[CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) for figures
and tables. All models and probes load from HuggingFace Hub — no local
checkpoints needed.

Dataset paths autodetect on crockett and Nibi; override with `ADE20K_ROOT` /
`IMAGENET_VAL` env vars otherwise (see `.envrc.example`).

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

# ImageNet-1K classification:
uv run python -m canvit_eval in1k-clf --mode finetuned

# Reconstruction quality (ablation checkpoints):
uv run python -m canvit_eval reconstruction \
    --model-repo canvit/canvitb16-abl-baseline-2026-03-02
```

### Batch — the full paper matrix

```bash
# Full paper matrix (5 seeds, sequential, single GPU):
uv run python -m canvit_eval.batch --n-runs 5

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
batches resume cleanly AND seed counts pool over time:

```bash
# Day 1: preliminary n=1 on a new grid.
uv run python -m canvit_eval.batch --tasks ade20k-seg --grids 9 --include-extra-grids --n-runs 1

# Day 2: add 4 more seeds without rerunning r=0. Export bootstraps over all 5.
uv run python -m canvit_eval.batch --tasks ade20k-seg --grids 9 --include-extra-grids --n-runs 5 --skip-existing
```

Each run writes `{stem}_{UTC_ts}_r{run}.pt`; the paper's export
(`export/ade20k_seg.py:_latest_per_run`) keeps the newest timestamp per run_idx,
so reruns don't contaminate old seeds if the config is unchanged.

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
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper (.pt → JSON → PDF) |
