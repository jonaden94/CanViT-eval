# CanViT-eval

Evaluation and benchmarking for CanViT. Produces `.pt` result files consumed by [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) for figures and tables.

All models and probes loaded from HuggingFace Hub. No local checkpoints needed.

Dataset paths are autodetected on known machines (crockett, nibi). Override with `ADE20K_ROOT` / `IMAGENET_VAL` env vars if needed. See `.envrc.example`.

## Usage

```bash
# Batch eval — all paper results in one command:
uv run python -m canvit_eval.batch --n-runs 1          # smoke test
uv run python -m canvit_eval.batch --n-runs 5          # full paper
uv run python -m canvit_eval.batch --tasks ade20k-seg   # single task
uv run python -m canvit_eval.batch --dry-run            # print commands (for SLURM)

# Individual evals:
uv run python -m canvit_eval ade20k-seg --probe-repo canvit/probe-ade20k-40k-s512-c32-in21k
uv run python -m canvit_eval in1k-clf
uv run python -m canvit_eval reconstruction --model-repo canvit/canvitb16-abl-...
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
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper (.pt → JSON → PDF) |
