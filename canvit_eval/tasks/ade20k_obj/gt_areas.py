"""Build per-(image, class) GT-area dataframe for ADE20K validation.

Outputs:
    AREA_PARQUET       one row per (image_idx, class_idx) with area (pixel fraction).
    AREA_STATS_PARQUET one row per class with mean/min/max area across the val set.

Both are consumed downstream by `iou` (this package) and by the mask-size figure
(plotting/figures/resolution_and_mask_size_analysis.py in CanViT-Toward-AVFMs).
"""

import logging
import time
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

# ADE20K 1-indexed class range (background = 0, classes 1..150 per the challenge).
MIN_CLASS_IDX = 1
MAX_CLASS_IDX = 150

# ADE20K validation set size (sanity-check anchor).
EXPECTED_N_VAL_IMAGES = 2000

_OBJECT_INFO_URL = (
    "http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016/objectInfo150.txt"
)


def _load_class_names(ade20k_root: Path) -> dict[int, str]:
    info_path = ade20k_root / "objectInfo150.txt"
    if not info_path.exists():
        log.info("objectInfo150.txt missing — downloading %s → %s", _OBJECT_INFO_URL, info_path)
        info_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_OBJECT_INFO_URL, info_path)
    names: dict[int, str] = {}
    with open(info_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            idx = int(parts[0].strip())
            names[idx] = parts[4].strip()
    assert len(names) == MAX_CLASS_IDX, f"expected {MAX_CLASS_IDX} class names, got {len(names)}"
    assert set(names) == set(range(MIN_CLASS_IDX, MAX_CLASS_IDX + 1)), sorted(names)[:5]
    log.info("loaded %d class names from %s", len(names), info_path)
    return names


def compute_class_area(ann: np.ndarray, class_idx: int) -> float:
    """Fraction of pixels in ann occupied by class_idx (1-indexed)."""
    return float((ann == class_idx).sum() / ann.size)


def build_image_class_dataframe(ade20k_root: Path) -> pd.DataFrame:
    """One row per (image, class) pair observed in the validation set.

    Columns:
        image_idx   int   0-based index into the sorted annotation file list.
        class_idx   int   1-based ADE20K class index (1..150).
        class_name  str   primary name of the class (first comma-separated entry).
        area        float fraction of image pixels occupied by the class.
    """
    ann_dir = ade20k_root / "annotations" / "validation"
    assert ann_dir.is_dir(), f"ADE20K val annotations not found at {ann_dir}"
    class_names = _load_class_names(ade20k_root)

    ann_paths = sorted(ann_dir.glob("*.png"))
    n_images = len(ann_paths)
    assert n_images == EXPECTED_N_VAL_IMAGES, (
        f"expected {EXPECTED_N_VAL_IMAGES} val images, found {n_images} in {ann_dir}"
    )
    log.info("scanning %d annotations under %s", n_images, ann_dir)

    rows: list[dict] = []
    skipped_bg = 0
    skipped_oob = 0
    for image_idx, ann_path in enumerate(ann_paths):
        arr = np.array(Image.open(ann_path))
        for cls in np.unique(arr).tolist():
            if cls == 0:
                skipped_bg += 1
                continue
            if cls < MIN_CLASS_IDX or cls > MAX_CLASS_IDX:
                skipped_oob += 1
                continue
            rows.append({
                "image_idx": image_idx,
                "class_idx": int(cls),
                "class_name": class_names[cls].split(",")[0],
                "area": compute_class_area(arr, cls),
            })
    df = pd.DataFrame(rows)
    assert len(df) > 0, "no (image, class) rows — ADE20K val set empty or unreadable?"
    assert df["class_idx"].between(MIN_CLASS_IDX, MAX_CLASS_IDX).all(), df["class_idx"].describe()
    assert df["image_idx"].max() == n_images - 1, (df["image_idx"].max(), n_images)
    log.info(
        "built %d (image, class) rows across %d classes "
        "(skipped %d bg occurrences, %d out-of-range)",
        len(df), df["class_idx"].nunique(), skipped_bg, skipped_oob,
    )
    return df


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

    root = ade20k_root()
    log.info("=== build area dataframe ===")
    log.info("ade20k_root=%s", root)
    log.info("area parquet     → %s", AREA_PARQUET)
    log.info("area stats parq  → %s", AREA_STATS_PARQUET)

    t0 = time.monotonic()
    flat_df = build_image_class_dataframe(root)
    stats_df = class_stats_dataframe(flat_df, root)
    log.info("built in %.1fs", time.monotonic() - t0)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    flat_df.to_parquet(AREA_PARQUET, compression="snappy")
    stats_df.to_parquet(AREA_STATS_PARQUET, compression="snappy")
    log.info(
        "wrote %s (%.1f MB, %d rows)",
        AREA_PARQUET, AREA_PARQUET.stat().st_size / 1e6, len(flat_df),
    )
    log.info(
        "wrote %s (%.1f KB, %d class rows)",
        AREA_STATS_PARQUET, AREA_STATS_PARQUET.stat().st_size / 1e3, len(stats_df),
    )


if __name__ == "__main__":
    main()
