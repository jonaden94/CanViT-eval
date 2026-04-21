# ADE20K Object Analysis

Steps to generate `dv3_ade20k_per_image.parquet` and `canvit_ade20k_per_image.parquet`, which are consumed by the CanViT-Toward-AVFMs repo.

## Step 1: Export DINOv3 patch features

Run `export_dv3_features.py` to extract DINOv3 features for ADE20K validation images. Features are saved to `output/dv3_features/{resolution}px_features.pt`.

To export all supported resolutions (128, 144, 160, 192, 256, 384, 512) in one go:

```bash
uv run python canvit_eval/tasks/ade20k_obj/export_dv3_features.py --all
```

Or export a single resolution with `--eval-resolution`:

```bash
uv run python canvit_eval/tasks/ade20k_obj/export_dv3_features.py --eval-resolution 256
```

## Step 2: Build the area dataframe

Run `dataframe_dataset.py` to compute per-(image, class) pixel-area statistics over the validation set. Outputs `output/ade20k_df_flat.parquet` and `output/ade20k_df_stats.parquet`.

```bash
uv run python canvit_eval/tasks/ade20k_obj/dataframe_dataset.py
```

## Step 3: Compute per-image IoU

Run `dataframe_iou_mask_size.py` to produce the final parquets.

**DINOv3** (uses pre-extracted features from Step 1):

```bash
uv run python canvit_eval/tasks/ade20k_obj/dataframe_iou_mask_size.py dinov3
```

Output: `output/dv3_ade20k_per_image.parquet`

**CanViT** (runs episodes on the fly):

To run all supported canvas resolutions (8, 9, 10, 12, 16, 24, 32, 64) in batches of 3:

```bash
uv run python canvit_eval/tasks/ade20k_obj/dataframe_iou_mask_size.py canvit --all
```

Or specify a subset manually:

```bash
uv run python canvit_eval/tasks/ade20k_obj/dataframe_iou_mask_size.py canvit \
    --canvas-resolutions 8 32 64
```

Output: `output/canvit_ade20k_per_image.parquet`

For DINOv3:
```bash
uv run python canvit_eval/tasks/ade20k_obj/dataframe_iou_mask_size.py dinov3
```
Output: `output/dv3_ade20k_per_image.parquet`
