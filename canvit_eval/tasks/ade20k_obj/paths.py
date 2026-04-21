"""Output paths for the ADE20K mask-size pipeline (SSOT)."""

from pathlib import Path


RESULTS_DIR = Path("./results")

# DINOv3 features (one .pt per input resolution, produced by export_dv3_features).
FEATURES_DIR = RESULTS_DIR / "dv3_features"

# Per-(image, class) area table — GT-only, produced by dataframe_dataset.
AREA_PARQUET = RESULTS_DIR / "ade20k_df_flat.parquet"
AREA_STATS_PARQUET = RESULTS_DIR / "ade20k_df_stats.parquet"

# Per-row IoU outputs of dataframe_iou_mask_size.
DV3_PARQUET = RESULTS_DIR / "dv3_ade20k_per_image.parquet"
CANVIT_PARQUET = RESULTS_DIR / "canvit_ade20k_per_image.parquet"


def features_path(eval_resolution_px: int) -> Path:
    return FEATURES_DIR / f"{eval_resolution_px}px_features.pt"
