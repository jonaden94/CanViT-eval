# ADE20K Mask Analysis

Produces per-(image, class) and per-(image, class, timestep) intersection/union counts for the ADE20K val set, feeding the object-size analysis figure: "when/why does CanViT beat DINOv3, by mask size?"

## Quickstart

```bash
uv run python -m canvit_eval.tasks.ade20k_obj
```

Runs the full pipeline in order, skipping any step whose outputs already exist in `results/`:

1. Export DINOv3 patch features for all resolutions → `results/dv3_features/`
2. Build per-(image, class) area dataframe → `results/ade20k_df_flat.parquet`
3. Compute per-(image, class) IoU for DINOv3 → `results/dv3_ade20k_per_image.parquet`
4. Compute per-(image, class, timestep) IoU for CanViT (all canvas resolutions) → `results/canvit_ade20k_per_image.parquet`

The outputs are consumed by CanViT-Toward-AVFMs for MASK-size LOWESS smoothing, and figure panels.

## Design rationale

**Raw counts, not IoU.** Parquets store `inter_px` and `union_px` (int64 pixel counts), not derived IoU floats. IoU is not reaggregatable: global mIoU = Σ(intersection) / Σ(union) across images, which differs from averaging per-image IoUs. Storing counts lets you compute both. It also avoids 0/0 NaNs when a class is absent from an image.

**Also store `gt_area_px`.** Needed for mask-size analysis. It is free to compute in the same pass since the target mask is already loaded.

**No binning in eval.** Raw counts are saved; binning (bin count, linear vs log, bin edges) is a figure-level decision handled downstream in CanViT-Toward-AVFMs. This avoids re-running eval every time the figure changes.

**Sanity check.** After running, verify: `mean over c of (Σ inter[:,c]) / (Σ union[:,c])` matches the headline mIoU from `ade20k_seg.py`. If it doesn't, something is wrong.

## Running individual steps

Each step can also be run standalone if needed.

**Step 1 — DINOv3 features** (`results/dv3_features/{resolution}px_features.pt`):

```bash
uv run python canvit_eval/tasks/ade20k_obj/export_dv3_features.py --all
uv run python canvit_eval/tasks/ade20k_obj/export_dv3_features.py --eval-resolution 256  # single resolution
```

**Step 2 — Area dataframe** (`results/ade20k_df_flat.parquet`, `results/ade20k_df_stats.parquet`):

```bash
uv run python canvit_eval/tasks/ade20k_obj/dataframe_dataset.py
```

**Step 3 — Per-(image, class) IoU:**

```bash
# DINOv3
uv run python canvit_eval/tasks/ade20k_obj/dataframe_iou_mask_size.py dinov3

# CanViT — all canvas resolutions (runs in subprocess batches to manage GPU memory)
uv run python canvit_eval/tasks/ade20k_obj/dataframe_iou_mask_size.py canvit --all

# CanViT — specific resolutions
uv run python canvit_eval/tasks/ade20k_obj/dataframe_iou_mask_size.py canvit \
    --canvas-resolutions 8 32 64
```
