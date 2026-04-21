"""Persistence utilities for ADE20K analysis outputs."""

import os
import urllib.request
from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field

_OBJECT_INFO_URL = (
    "http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016/objectInfo150.txt"
)

from PIL import Image
import pandas as pd
import numpy as np


ResizeMode = Literal["center_crop", "squish"]
OUTPUT_DIR = Path("./results")


def ade20k_root() -> Path:
    if root := os.environ.get("ADE20K_ROOT"):
        return Path(root)
    if tmpdir := os.environ.get("SLURM_TMPDIR"):
        return Path(tmpdir) / "ADEChallengeData2016"
    raise ValueError("ADE20K_ROOT env var not set and not running under SLURM")


@dataclass
class Config:
    probe_repo: str
    ade20k_root: Path = field(default_factory=ade20k_root)
    output: Path = Path("output/ade20k_seg.pt")
    scene_size: int = 512
    resize_mode: ResizeMode = "squish"
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True


def _load_class_names(cfg: Config) -> dict[int, str]:
    info_path = cfg.ade20k_root / "objectInfo150.txt"
    print(info_path)
    if not info_path.exists():
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


def build_image_class_dataframe(cfg: Config) -> pd.DataFrame:
    """One row per (image, class) pair observed in the validation set.

    Columns:
        image_idx   int   0-based index into the sorted annotation file list
        class_idx   int   1-based ADE20K class index
        class_name  str   primary name of the class (first comma-separated entry)
        area        float fraction of image pixels occupied by the class
    """
    ann_dir = cfg.ade20k_root / "annotations" / "validation"
    class_names = _load_class_names(cfg)
    rows = []
    for image_idx, ann_path in enumerate(sorted(ann_dir.glob("*.png"))):
        arr = np.array(Image.open(ann_path))
        for cls in np.unique(arr).tolist():
            if cls == 0 or cls > 150:
                continue
            mask = arr == cls
            area = compute_class_area(arr, cls)
            rows.append({
                "image_idx": image_idx,
                "class_idx": cls,
                "class_name": class_names[cls].split(",")[0],
                "area": area,
            })
    return pd.DataFrame(rows)


def class_area_range(flat_df: pd.DataFrame, class_idx: int) -> tuple[float, float]:
    """Return (min_area, max_area) across all validation images for class_idx."""
    areas = flat_df.loc[flat_df["class_idx"] == class_idx, "area"]
    if areas.empty:
        return (0.0, 0.0)
    return float(areas.min()), float(areas.max())


def class_stats_dataframe(flat_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aggregate flat_df to one row per class: mean_area, max_area, min_area."""
    class_names = _load_class_names(cfg)
    stats = flat_df.groupby("class_idx").agg(
        mean_area=("area", "mean"),
        max_area=("area", "max"),
        min_area=("area", "min"),
    )
    stats["name"] = stats.index.map(lambda i: class_names[i].split(",")[0])
    return stats.sort_values("mean_area", ascending=False)

if __name__ == "__main__":
    os.environ.setdefault("ADE20K_ROOT", "./data/ADEChallengeData2016")
    cfg = Config(probe_repo="canvit-probes")

    flat_df = build_image_class_dataframe(cfg)
    df = class_stats_dataframe(flat_df, cfg)

    flat_df.to_parquet(OUTPUT_DIR / "ade20k_df_flat.parquet", compression='snappy')
    df.to_parquet(OUTPUT_DIR / "ade20k_df_stats.parquet", compression='snappy')
