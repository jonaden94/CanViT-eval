# CanViT-eval

Evaluation and benchmarking for CanViT. Produces `.pt` result files consumed by [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) for figures and tables.

All models and probes loaded from HuggingFace Hub. No local checkpoints needed.

## Usage

```bash
# ADE20K segmentation — CanViT (multi-timestep)
ADE20K_ROOT=/path/to/ADE20k uv run python -m canvit_eval ade20k-seg \
    --probe-repo canvit/probe-ade20k-40k-s512-c32-in21k

# ADE20K segmentation — DINOv3 baseline (single-pass)
ADE20K_ROOT=/path/to/ADE20k uv run python -m canvit_eval ade20k-seg \
    --model dinov3 --probe-repo canvit/probe-ade20k-40k-dv3b-128px --eval-resolution 128

# Batch eval (all policies × resolutions × runs)
ADE20K_ROOT=/path/to/ADE20k uv run python -m canvit_eval.batch --n-runs 5

# IN1K classification
uv run python -m canvit_eval in1k-clf

# Reconstruction quality (see analysis/ablations in paper repo for model repo IDs)
ADE20K_ROOT=/path/to/ADE20k uv run python -m canvit_eval reconstruction \
    --model-repo canvit/canvitb16-abl-<variant>-<date>
```

## Architecture

```bash
uv run pypatree
```

## Dependencies

- [canvit](https://github.com/yberreby/CanViT-PyTorch-Next) (private) — core model + policies
- [canvit-probes](https://github.com/m2b3/CanViT-probes) — probe definitions, datasets, metrics

## Tests

```bash
uv run pytest tests/ -v
```

## Related repos

| Repo | Role |
|------|------|
| [CanViT-PyTorch-Next](https://github.com/yberreby/CanViT-PyTorch-Next) | Core model (private) |
| [CanViT-probes](https://github.com/m2b3/CanViT-probes) | Probes, datasets, metrics, training |
| [CanViT-pretrain](https://github.com/m2b3/CanViT-pretrain) | Model pretraining |
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper (.pt → JSON → PDF) |
