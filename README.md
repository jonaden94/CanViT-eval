# CanViT-eval

Evaluation and benchmarking for CanViT. Produces `.pt` result files consumed by [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) for figures and tables.

All models and probes loaded from HuggingFace Hub. No local checkpoints needed.

## Design

**One eval pipeline per task, shared across models.** For ADE20K segmentation, the same dataset loading, probe application, and IoU computation handles both CanViT (multi-timestep) and DINOv3 (single-pass). Only the feature extractor differs.

```
canvit_eval/
    episode.py       # Shared CanViT forward loop (run_episode)
    features.py      # Feature extractors (CanViT + DINOv3)
    policies.py      # Interactive policies (wraps canvit.policies)
    metrics.py       # IoU accumulator
    config.py        # Shared defaults (model repo, ADE20K root)
    datasets/        # ADE20K, IN1K dataset loading
    tasks/           # Per-task eval (ade20k_seg, in1k_clf, reconstruction)
    __main__.py      # Unified CLI
```

## Usage

```bash
# ADE20K segmentation — CanViT (multi-timestep)
ADE20K_ROOT=/path/to/ADE20k uv run python -m canvit_eval ade20k-seg \
    --probe-repo canvit/probe-ade20k-40k-s512-c32-in21k

# ADE20K segmentation — DINOv3 baseline (single-pass)
ADE20K_ROOT=/path/to/ADE20k uv run python -m canvit_eval ade20k-seg \
    --model dinov3 --probe-repo canvit/probe-ade20k-40k-dv3b-128px --eval-resolution 128

# IN1K classification
uv run python -m canvit_eval in1k-clf

# Reconstruction quality (ablation checkpoints)
ADE20K_ROOT=/path/to/ADE20k uv run python -m canvit_eval reconstruction \
    --model-repo canvit/canvitb16-abl-baseline-2026-03-15
```

## Dependencies

- [canvit](https://github.com/m2b3/CanViT-PyTorch) — core model + policies
- [canvit-probes](https://github.com/m2b3/CanViT-probes) — probe definitions (SegmentationProbe)

## Tests

```bash
ADE20K_ROOT=/fake uv run pytest tests/ -v --cov=canvit_eval --cov-report=term-missing
```

15 tests, 92% coverage on core. Integration tests with real HF model + probe.

## Related repos

| Repo | Role |
|------|------|
| [CanViT-PyTorch](https://github.com/m2b3/CanViT-PyTorch) | Core model + policies |
| [CanViT-probes](https://github.com/m2b3/CanViT-probes) | Probe definitions |
| [CanViT-train](https://github.com/m2b3/CanViT-train) | Pretraining + probe training |
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper (.pt → JSON → PDF) |
