"""Build per-(image, class) GT-area dataframe for ADE20K validation.

Outputs:
    AREA_PARQUET       one row per (image_idx, class_idx) with area (pixel fraction).
    AREA_STATS_PARQUET one row per class with mean/min/max area across the val set.

Both are consumed downstream by `iou` (this package) and by the mask-size figure
(plotting/figures/resolution_and_mask_size_analysis.py in CanViT-Toward-AVFMs).
"""

import json
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from canvit_eval.config import ade20k_root
from canvit_eval.provenance import provenance
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


# Raw ADE20K PNGs encode class index in pixel values 0..150 (0 = background,
# 1..150 = semantic classes). uint8 bincount has 256 bins exactly.
_BINCOUNT_BINS = 256


def build_image_class_dataframe(ade20k_root: Path) -> pd.DataFrame:
    """One row per (image, class) pair observed in the validation set.

    Columns:
        image_idx   int   0-based index into the sorted annotation file list.
        class_idx   int   1-based ADE20K class index (1..150).
        class_name  str   primary name of the class (first comma-separated entry).
        area        float fraction of image pixels occupied by the class.

    Vectorised: `np.bincount(arr.ravel(), minlength=256)` returns counts for
    all pixel values in ONE pass. Old code did `(arr == c).sum()` per class
    observed — N_classes passes over the mask per image.
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
    skipped_bg_pixels = 0
    skipped_oob_pixels = 0
    for image_idx, ann_path in enumerate(ann_paths):
        arr = np.array(Image.open(ann_path))
        counts = np.bincount(arr.ravel(), minlength=_BINCOUNT_BINS)
        size = arr.size
        skipped_bg_pixels += int(counts[0])
        # Anything outside [MIN_CLASS_IDX, MAX_CLASS_IDX] beyond background.
        if counts.size > MAX_CLASS_IDX + 1:
            skipped_oob_pixels += int(counts[MAX_CLASS_IDX + 1:].sum())
        for cls in range(MIN_CLASS_IDX, MAX_CLASS_IDX + 1):
            n = int(counts[cls])
            if n == 0:
                continue
            rows.append({
                "image_idx": image_idx,
                "class_idx": cls,
                "class_name": class_names[cls].split(",")[0],
                # Direct divide (not `n * (1/size)`): arr.size is not generally
                # a power of 2 on raw ADE20K PNGs, and reciprocal-multiply
                # diverges from divide in the last ULP. Match the per-class
                # reference exactly.
                "area": n / size,
            })
    df = pd.DataFrame(rows)
    assert len(df) > 0, "no (image, class) rows — ADE20K val set empty or unreadable?"
    assert df["class_idx"].between(MIN_CLASS_IDX, MAX_CLASS_IDX).all(), df["class_idx"].describe()
    assert df["image_idx"].max() == n_images - 1, (df["image_idx"].max(), n_images)
    log.info(
        "built %d (image, class) rows across %d classes "
        "(background pixels: %d, out-of-range pixels: %d)",
        len(df), df["class_idx"].nunique(), skipped_bg_pixels, skipped_oob_pixels,
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
    AREA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
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
    sidecar = AREA_PARQUET.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "stage": "gt_areas",
        "ade20k_root": str(root),
        "n_images": int(flat_df["image_idx"].nunique()),
        "n_rows": len(flat_df),
        "n_classes_with_data": int(flat_df["class_idx"].nunique()),
        **provenance(),
    }, indent=2))
    log.info("sidecar → %s", sidecar)


if __name__ == "__main__":
    main()
