# ADE20K Mask Analysis

Produces per-(image, class) and per-(image, class, timestep) intersection/union counts for the ADE20K val set, feeding the mask-size figure in CanViT-Toward-AVFMs: "when/why does CanViT beat DINOv3, by mask size?"

Output paths are defined once in `paths.py` (SSOT). Append `--help` to any command below for the full flag set.

## Design rationale

**Raw counts, not IoU.** Parquets store `inter_px` and `union_px` (int64 pixel counts), not derived IoU floats. IoU is not reaggregatable: global mIoU = Σ(intersection) / Σ(union) across images, which differs from averaging per-image IoUs. Storing counts lets you compute both, and avoids 0/0 NaNs when a class is absent from an image.

**Also store `gt_area_px`.** Needed for mask-size binning downstream. Free to compute in the same pass since the target mask is already loaded.

**No binning in eval.** Raw counts are saved; binning (bin count, linear vs log, bin edges) is a figure-level decision in CanViT-Toward-AVFMs. Eval does not re-run when the figure's bins change.

**Sanity check.** After running, `mean over c of (Σ inter[:,c]) / (Σ union[:,c])` should match the headline mIoU from `ade20k_seg.py`. Divergence = bug.

## Run the whole pipeline

```bash
uv run python -m canvit_eval.tasks.ade20k_obj
```

Runs all four stages in order. Each stage is skipped if its output already exists — the size + mtime of the skipped artifact are logged so you can sanity-check what's being reused. To force a stage to rerun, delete its output.

The orchestrator defaults to the canvas resolutions the paper figure consumes (`{8, 16, 32, 64}`) and the single DINOv3 input resolution it uses (128 px). Adjust the lists at the top of `__main__.py` to expand the sweep.

## Or run each stage standalone

**1. DINOv3 features** — `FEATURES_DIR / "{eval_resolution_px}px_features.pt"`

```bash
uv run python -m canvit_eval.tasks.ade20k_obj.export_dv3_features --eval-resolution-px 128
```

**2. Area dataframe** — `AREA_PARQUET`, `AREA_STATS_PARQUET`

```bash
uv run python -m canvit_eval.tasks.ade20k_obj.gt_areas
```

**3. Per-image IoU** — DINOv3 and CanViT are independent subcommands.

```bash
# DINOv3 — consumes every features.pt present under FEATURES_DIR.
uv run python -m canvit_eval.tasks.ade20k_obj.iou dinov3

# CanViT — pass the canvas resolutions you want. Each runs in its own
# subprocess so CUDA memory is fully released between them.
uv run python -m canvit_eval.tasks.ade20k_obj.iou canvit \
    --canvas-resolutions 8 16 32 64
```

Valid canvas resolutions are the keys of `CANVIT_PROBE_REPOS` in `iou.py`.
