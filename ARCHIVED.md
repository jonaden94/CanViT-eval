# CanViT-eval is ARCHIVED (2026-09-02) — read-only reference

Its functionality lives in **CanViT-train** (`python -m canvit_train.harness.evaluate`) and
**CanViT-PyTorch** (`bench/pt/`). Do not edit this repo, and do not run it in preference to
the merged version. Full record: `CanViT-train/unification_docs/20-eval-merge.md`.

| was here | is now |
|---|---|
| `ade20k-seg-canvit` | `harness.evaluate ade20k` |
| `ade20k-seg-dinov3` | `harness.evaluate ade20k-dinov3` |
| `in1k-clf` | `harness.evaluate in1k` |
| `reconstruction` | `harness.evaluate distill` (+ `--cfg.val-image-dir` for a flat folder) |
| `tests/test_iou_equivalence.py` | `canvit_train/ade20k/test_per_row_iou.py` |
| `tests/test_view_scale.py` | `canvit_train/harness/tests/test_evaluate.py` (ported by case) |
| `bench/pt/` | `CanViT-PyTorch/bench/pt/` |
| `batch.py`, `ade20k_obj/`'s staged pipeline, `slurm_nhr/` | **not ported** — see §3 of the doc |

Where both survived to be compared, they agree exactly: ADE20K over ten timesteps and IN1k
top-1/top-5 came out bit-identical between the two repos on real checkpoints.

## ⚠️ `results/` — two of these numbers are wrong

**`reconstruction`'s `scene_cos_raw` and `cls_cos_raw` are understated.** This repo compared
the model's prediction, which lives in the teacher-*normalized* space, against *raw* teacher
features, with no destandardization anywhere in `tasks/reconstruction.py`. Measured on
exp32-fovi step-1916928 at t9: **0.648 here vs 0.927 correct** — a 0.279 error. Confirmed
from the tensors, not just by reading: the prediction's per-channel moments (std 0.849,
‖vec‖ 23.58) match the normalized teacher (0.988 / 27.39), not the raw one (0.329 / 11.58),
and both mismatched pairings come out low while both matched ones come out high.

`scene_cos_norm` / `cls_cos_norm` compare like with like and are **fine**.

CanViT-train's `validate` always destandardized, so every distill run's logged
`val/scene_cos_raw_*` — exp22 onward — is the correct number. Only this repo's separate
reconstruction task crossed the spaces.

**`ade20k_obj` outputs** came from a task that imports `pandas` at module level while
`pandas` sits in this repo's *dev* dependency group, so it only ever ran in a dev install.
Treat any results from it as unverified.

## Other defects found while merging, fixed elsewhere

- `bench/pt/run.py` called `CanViT.forward(glimpse=...)`; bare `CanViT` takes `image=`. Every
  CanViT cell raised `TypeError`, so the benchmark had not run against current core for some
  time. Fixed in `CanViT-PyTorch/bench/pt/`.
- `random` means something different here than in CanViT-train: this repo's does not open on
  the full-scene anchor and is not patcher-aware, a 0.0205 difference at t0 on a foveated
  model. For a foveated model CanViT-train's is the correct one. Both are reachable there —
  see the `t0` modifier.
