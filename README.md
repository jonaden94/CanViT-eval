# CanViT-eval

Evaluation and benchmarking for CanViT. Produces `.pt` result files consumed by [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) for figures and tables.

## Design

**One eval pipeline per task, shared across models.** For ADE20K segmentation, the same dataset loading, probe application, and IoU computation is used for both CanViT (multi-timestep episode) and DINOv3 (single passive forward pass). The only thing that differs is how features are extracted — a `FeatureExtractor` callable.

```
canvit_eval/
    episode.py       # THE shared CanViT forward loop
    features.py      # CanViT + DINOv3 feature extractors
    policies.py      # 6 viewing policies
    metrics.py       # IoU accumulator
    config.py        # Shared config (model repo, ADE20K root, etc.)
    evaluate.py      # Generic eval function (MetricAccumulator protocol)
    tasks/
        ade20k_seg.py       # ADE20K segmentation (probe + IoU)
        in1k_clf.py         # IN1K classification (clf probe + top-1)
        reconstruction.py   # Cosine sim to DINOv3 teacher
    datasets/
        ade20k.py    # ADE20K dataset + transforms
        imagenet.py  # IN1K dataset
    __main__.py      # CLI: ade20k-seg | in1k-clf | reconstruction
```

## Usage

```bash
# ADE20K segmentation — CanViT
ADE20K_ROOT=/path/to/ADE20k uv run python -m canvit_eval ade20k-seg \
    --model canvit --cfg.probe-repo canvit/probe-ade20k-40k-s512-c32-in21k

# ADE20K segmentation — DINOv3 baseline
ADE20K_ROOT=/path/to/ADE20k uv run python -m canvit_eval ade20k-seg \
    --model dinov3 --cfg.probe-repo canvit/probe-ade20k-40k-dv3b-128px

# IN1K classification
uv run python -m canvit_eval in1k-clf

# Reconstruction quality
ADE20K_ROOT=/path/to/ADE20k uv run python -m canvit_eval reconstruction \
    --model-repo canvit/canvitb16-abl-baseline-2026-03-15
```

## Tests

```bash
uv run pytest tests/ -v --cov=canvit_eval --cov-report=term-missing
```

15 tests, 92% coverage on core modules. Integration tests use real HF model.
