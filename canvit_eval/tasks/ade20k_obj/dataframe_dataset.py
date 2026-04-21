"""Build per-(image, class) GT-area dataframe for ADE20K validation.

Outputs:
    AREA_PARQUET       one row per (image_idx, class_idx) with area (pixel fraction).
    AREA_STATS_PARQUET one row per class with mean/min/max area across the val set.

Both are consumed downstream by `dataframe_iou_mask_size` and by figure generators.
"""

import logging
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from canvit_eval.config import ade20k_root
from canvit_eval.tasks.ade20k_obj.paths import (
    AREA_PARQUET,
    AREA_STATS_PARQUET,
    RESULTS_DIR,
)

log = logging.getLogger(__name__)

_OBJECT_INFO_URL = (
    "http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016/objectInfo150.txt"
)


def _load_class_names(ade20k_root: Path) -> dict[int, str]:
    info_path = ade20k_root / "objectInfo150.txt"
    if not info_path.exists():
        log.info("downloading %s → %s", _OBJECT_INFO_URL, info_path)
        info_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_OBJECT_INFO_URL, info_path)
    names = {}
    with open(info_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            idx = int(parts[0].strip())
            names[idx] = parts[4].strip()
    return names


def compute_class_area(ann: np.ndarray, class_idx: int) -> float:
    """Fraction of pixels in ann occupied by class_idx (1-indexed)."""
    return float((ann == class_idx).sum() / ann.size)


def build_image_class_dataframe(ade20k_root: Path) -> pd.DataFrame:
    """One row per (image, class) pair observed in the validation set.

    Columns:
        image_idx   int   0-based index into the sorted annotation file list
        class_idx   int   1-based ADE20K class index
        class_name  str   primary name of the class (first comma-separated entry)
        area        float fraction of image pixels occupied by the class
    """
    ann_dir = ade20k_root / "annotations" / "validation"
    class_names = _load_class_names(ade20k_root)
    rows = []
    for image_idx, ann_path in enumerate(sorted(ann_dir.glob("*.png"))):
        arr = np.array(Image.open(ann_path))
        for cls in np.unique(arr).tolist():
            if cls == 0 or cls > 150:
                continue
            rows.append({
                "image_idx": image_idx,
                "class_idx": cls,
                "class_name": class_names[cls].split(",")[0],
                "area": compute_class_area(arr, cls),
            })
    return pd.DataFrame(rows)


def class_area_range(flat_df: pd.DataFrame, class_idx: int) -> tuple[float, float]:
    """Return (min_area, max_area) across all validation images for class_idx."""
    areas = flat_df.loc[flat_df["class_idx"] == class_idx, "area"]
    if areas.empty:
        return (0.0, 0.0)
    return float(areas.min()), float(areas.max())


def class_stats_dataframe(flat_df: pd.DataFrame, ade20k_root: Path) -> pd.DataFrame:
    """Aggregate flat_df to one row per class: mean_area, max_area, min_area."""
    class_names = _load_class_names(ade20k_root)
    stats = flat_df.groupby("class_idx").agg(
        mean_area=("area", "mean"),
        max_area=("area", "max"),
        min_area=("area", "min"),
    )
    stats["name"] = stats.index.map(lambda i: class_names[i].split(",")[0])
    return stats.sort_values("mean_area", ascending=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    os.environ.setdefault("ADE20K_ROOT", "./data/ADEChallengeData2016")
    root = ade20k_root()
    log.info("ade20k_root=%s", root)
    log.info("will write: %s", AREA_PARQUET)
    log.info("will write: %s", AREA_STATS_PARQUET)

    flat_df = build_image_class_dataframe(root)
    stats_df = class_stats_dataframe(flat_df, root)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    flat_df.to_parquet(AREA_PARQUET, compression="snappy")
    stats_df.to_parquet(AREA_STATS_PARQUET, compression="snappy")
    log.info("done: %d (image, class) rows", len(flat_df))


if __name__ == "__main__":
    main()
