"""Full ADE20K object-size pipeline.

Runs all three steps in order, skipping steps whose outputs already exist:
  1. Export DINOv3 patch features for all resolutions → results/dv3_features/
  2. Build per-(image, class) area dataframe           → results/ade20k_df_flat.parquet
  3. Compute per-image IoU for DINOv3 and CanViT       → results/dv3_ade20k_per_image.parquet
                                                          results/canvit_ade20k_per_image.parquet

Usage:
    uv run python -m canvit_eval.tasks.ade20k_obj
"""

import logging
from pathlib import Path

from canvit_eval.tasks.ade20k_obj.dataframe_dataset import (
    Config as DatasetConfig,
    build_image_class_dataframe,
    class_stats_dataframe,
    OUTPUT_DIR,
)
from canvit_eval.tasks.ade20k_obj.export_dv3_features import (
    ALL_RESOLUTIONS,
    Config as FeaturesConfig,
    main as export_features,
)
from canvit_eval.tasks.ade20k_obj.dataframe_iou_mask_size import (
    DEFAULT_RESULTS_DIR,
    DINOv3Config,
    CanViTConfig,
    run_dinov3,
    run_canvit,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEATURES_DIR = DEFAULT_RESULTS_DIR / "dv3_features"
AREA_PARQUET = DEFAULT_RESULTS_DIR / "ade20k_df_flat.parquet"


# ── Step 1: DINOv3 features ───────────────────────────────────────────────────

missing = [r for r in ALL_RESOLUTIONS if not (FEATURES_DIR / f"{r}px_features.pt").exists()]
if not missing:
    log.info("Step 1: all DINOv3 features already present, skipping.")
else:
    log.info("Step 1: exporting DINOv3 features for resolutions %s", missing)
    cfg = FeaturesConfig()
    for res in missing:
        cfg.eval_resolution = res
        export_features(cfg)


# ── Step 2: Area dataframe ────────────────────────────────────────────────────

if AREA_PARQUET.exists():
    log.info("Step 2: area parquet already present, skipping.")
else:
    log.info("Step 2: building area dataframe")
    cfg = DatasetConfig(probe_repo="")
    flat_df = build_image_class_dataframe(cfg)
    stats_df = class_stats_dataframe(flat_df, cfg)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    flat_df.to_parquet(OUTPUT_DIR / "ade20k_df_flat.parquet", compression="snappy")
    stats_df.to_parquet(OUTPUT_DIR / "ade20k_df_stats.parquet", compression="snappy")
    log.info("Saved %s", OUTPUT_DIR / "ade20k_df_flat.parquet")


# ── Step 3a: DINOv3 per-image IoU ─────────────────────────────────────────────

dv3_out = DEFAULT_RESULTS_DIR / "dv3_ade20k_per_image.parquet"
if dv3_out.exists():
    log.info("Step 3a: DINOv3 parquet already present, skipping.")
else:
    log.info("Step 3a: computing DINOv3 per-image IoU")
    run_dinov3(DINOv3Config())


# ── Step 3b: CanViT per-image IoU ─────────────────────────────────────────────

canvit_out = DEFAULT_RESULTS_DIR / "canvit_ade20k_per_image.parquet"
if canvit_out.exists():
    log.info("Step 3b: CanViT parquet already present, skipping.")
else:
    log.info("Step 3b: computing CanViT per-image IoU (all canvas resolutions)")
    run_canvit(CanViTConfig(all=True))


log.info("Pipeline complete. Outputs in %s", DEFAULT_RESULTS_DIR)
