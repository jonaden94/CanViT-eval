# CanViT-eval

Evaluation and benchmarking for CanViT.

## Tasks

- **`ade20k-seg-canvit`** — ADE20K semantic segmentation via a CanViT episode
  (T-step rollout, mIoU per timestep).
- **`ade20k-seg-dinov3`** — same task with a passive DINOv3 backbone (single
  forward, t=0 only).
- **`in1k-clf`** — ImageNet-1k top-k classification via
  `CanViTForImageClassification`, either fused frozen probe or finetuned
  weights.
- **`reconstruction`** — cosine similarity between CanViT canvas / CLS and
  DINOv3 teacher features per timestep.

## Install

    uv sync

## Dataset paths

Set the env vars in `.envrc.example`, or pass paths via the per-task CLI flags:

    ADE20K_ROOT     # path to ADEChallengeData2016
    IMAGENET_VAL    # path to ILSVRC2012/val

## Single eval

    uv run python -m canvit_eval <subcommand> --help

Each subcommand is a tyro Config dataclass; `--help` lists every field with
its default and type.

## Batch eval

Build and run the eval matrix sequentially on a single GPU:

    uv run python -m canvit_eval.batch --help

`--skip-existing` resumes an interrupted batch (matched on structural identity,
not filename).

## ADE20K mask-size pipeline

Per-(image, class, timestep) IoU. Three stages — DINOv3 feature export, DINOv3
IoU, CanViT IoU. Each stage skips if its output already exists:

    uv run python -m canvit_eval.tasks.ade20k_obj

Stages can also be invoked individually; see `--help` on each module.

## Latency bench

Per-forward-pass latency at `batch_size=1` with explicit device sync. `matrix.py`
generates one subprocess per cell (pre-flight gated on GPU/CPU idle) and writes
one JSONL per cell; `analyze.py` reports distributional stats over the JSONLs:

    uv run python bench/pt/matrix.py --help
    uv run python bench/pt/run.py --help
    uv run python bench/pt/analyze.py --pattern 'bench/pt/results/*.jsonl'

## Tests

    uv run pytest
