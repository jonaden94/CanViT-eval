"""Output paths for the ADE20K mask-size pipeline (SSOT).

Matches the `results/{task_dir}/` convention used by `canvit_eval.batch` —
each task namespaces its artifacts under its own subdir. Filenames inside
the subdir drop redundant prefixes (the "ade20k_obj" is in the path; no need
to repeat it in each filename).
"""

from pathlib import Path


RESULTS_DIR = Path("./results")

# All ade20k_obj artifacts live under one subdir so they rsync together and
# can be rm'd/archived as a unit.
TASK_DIR = RESULTS_DIR / "ade20k_obj"

# DINOv3 features (one .pt per input resolution, produced by export_dv3_features).
FEATURES_DIR = TASK_DIR / "dv3_features"

# Per-row IoU outputs of `iou` (consumed by the paper's mask-size figure).
# Full N×C cross product per (resolution / canvas, timestep) — gt_area_px is
# emitted directly per row, so no separate areas table is needed.
DV3_PARQUET = TASK_DIR / "dv3_iou.parquet"
CANVIT_PARQUET = TASK_DIR / "canvit_iou.parquet"


def features_path(eval_resolution_px: int) -> Path:
    return FEATURES_DIR / f"{eval_resolution_px}px_features.pt"


# ADE20K val split — used as a sanity-check denominator across the pipeline.
EXPECTED_N_VAL_IMAGES = 2000
