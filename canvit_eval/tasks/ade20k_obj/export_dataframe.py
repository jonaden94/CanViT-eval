"""Per-(image, class) IoU for DINOv3 on ADE20K validation set.

Loads pre-extracted features from ../explore-ade20k/outputs/{res}px/features.pt,
runs the matching linear probe, and computes per-(image, class) IoU via the
histc confusion matrix trick. Merges with the (image_idx, class_idx, area)
dataframe and saves a long-form parquet.

Output columns:
    image_idx    int   0-based index into the val set
    class_idx    int   1-based ADE20K class (1–150)
    resolution   int   eval resolution in pixels
    iou          f32   per-image per-class IoU (only rows where class is present)
    gt_area_px   i32   GT pixel count for this class in this image
    area         f64   fraction of image pixels (from ade20k_df_flat.parquet)
    class_name   str   primary class name (from ade20k_df_flat.parquet)

Usage:
    uv run python canvit_eval/tasks/ade20k_obj/export_dataframe.py
    uv run python canvit_eval/tasks/ade20k_obj/export_dataframe.py --resolutions 512
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from canvit_pytorch import SegmentationProbe
from canvit_specialize.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES
from tqdm import tqdm
import tyro

log = logging.getLogger(__name__)

PROBE_REPO_TEMPLATE = "canvit/probe-ade20k-40k-dv3b-{resolution}px"
DEFAULT_EXPORTS_DIR = Path(__file__).parents[4] / "explore-ade20k" / "outputs"
DEFAULT_AREA_PARQUET = DEFAULT_EXPORTS_DIR / "ade20k_df_flat.parquet"


@dataclass
class Config:
    exports_dir: Path = DEFAULT_EXPORTS_DIR
    """Directory containing {res}px/features.pt subdirectories."""
    area_parquet: Path = DEFAULT_AREA_PARQUET
    """Path to ade20k_df_flat.parquet with (image_idx, class_idx, area, class_name)."""
    resolutions: list[int] = field(default_factory=lambda: [128, 144, 160, 192, 256, 384, 512])
    """Eval resolutions to process. Must have a matching features.pt file."""
    output: Path = Path("output/dv3_ade20k_per_image.parquet")
    batch_size: int = 32
    device: str = "auto"


def _get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _per_image_iou(
    pred: torch.Tensor,   # [H, W]  int64, 0-indexed class predictions
    mask: torch.Tensor,   # [H, W]  int64, 0-indexed GT (255 = ignore)
    C: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Histc confusion matrix trick: returns (inter, union, gt_area), each [C].

    cm[pred_class, gt_class] = pixel count.
    - intersection = diagonal (TP per class)
    - union = row_sum + col_sum - diagonal (TP + FP + FN)
    - gt_area = col_sum (total GT pixels per class)
    """
    valid = mask != IGNORE_LABEL
    p = pred[valid]
    t = mask[valid]
    cm = torch.histc((p * C + t).float(), bins=C * C, min=0, max=C * C - 1)
    cm = cm.reshape(C, C)
    inter = cm.diag()
    union = cm.sum(1) + cm.sum(0) - inter
    gt_area = cm.sum(0)
    return inter, union, gt_area


@torch.inference_mode()
def process_resolution(
    resolution: int,
    cfg: Config,
    device: torch.device,
) -> pd.DataFrame:
    """Run probe over features.pt for one resolution; return long-form DataFrame."""
    features_path = cfg.exports_dir / f"{resolution}px" / "features.pt"
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    data = torch.load(features_path, map_location="cpu", weights_only=False)
    feats_all: torch.Tensor = data["feats"]   # [N, grid*grid, D]
    masks_all: torch.Tensor = data["masks"]   # [N, H, W]  uint8
    grid: int = data["grid"]
    N = feats_all.shape[0]

    probe_repo = PROBE_REPO_TEMPLATE.format(resolution=resolution)
    probe = SegmentationProbe.from_pretrained(probe_repo).to(device).eval()
    log.info("res=%dpx  N=%d  grid=%d  probe=%s", resolution, N, grid, probe_repo)

    C = NUM_CLASSES
    inter_all = torch.zeros(N, C, dtype=torch.float32)
    union_all = torch.zeros(N, C, dtype=torch.float32)
    gt_area_all = torch.zeros(N, C, dtype=torch.float32)

    for start in tqdm(range(0, N, cfg.batch_size), desc=f"{resolution}px", leave=False):
        end = min(start + cfg.batch_size, N)
        feats = feats_all[start:end].to(device)                    # [B, grid*grid, D]
        masks = masks_all[start:end].to(device).long()             # [B, H, W]
        B = feats.shape[0]

        feats = feats.view(B, grid, grid, -1)                      # [B, grid, grid, D]
        logits = probe(feats.float())                              # [B, C, grid, grid]
        if logits.shape[-1] != masks.shape[-1]:
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False,
            )
        preds = logits.argmax(dim=1)                               # [B, H, W]

        for i in range(B):
            inter, union, gt_area = _per_image_iou(preds[i], masks[i], C)
            inter_all[start + i] = inter.cpu()
            union_all[start + i] = union.cpu()
            gt_area_all[start + i] = gt_area.cpu()

    # Sanity check: global mIoU must match canvit_eval reference
    total_inter = inter_all.sum(0)   # [C]
    total_union = union_all.sum(0)   # [C]
    valid_cls = total_union > 0
    global_miou = (total_inter[valid_cls] / total_union[valid_cls]).mean().item()
    log.info("  sanity mIoU @ %dpx = %.2f%%  (expect ~%.1f%%)",
             resolution, 100 * global_miou, {128: 34.2, 144: 36.4, 160: 38.3,
             192: 40.5, 256: 43.5, 384: 46.0, 512: 47.2}.get(resolution, float("nan")))

    # Convert to long-form: one row per (image, class) where class is present
    iou_vals = inter_all / (union_all + 1e-8)   # [N, C]

    rows = []
    for img_idx in range(N):
        present = union_all[img_idx] > 0         # [C] bool
        cls_indices = present.nonzero(as_tuple=True)[0]
        for c in cls_indices.tolist():
            rows.append({
                "image_idx": img_idx,
                "class_idx": c + 1,              # 0-indexed → 1-indexed (matches parquet)
                "resolution": resolution,
                "iou": float(iou_vals[img_idx, c]),
                "gt_area_px": int(gt_area_all[img_idx, c]),
            })

    return pd.DataFrame(rows)


def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    device = _get_device(cfg.device)
    log.info("device=%s  resolutions=%s", device, cfg.resolutions)

    area_df = pd.read_parquet(cfg.area_parquet, columns=["image_idx", "class_idx", "area", "class_name"])
    log.info("area_df: %d rows", len(area_df))

    frames = []
    for res in cfg.resolutions:
        df = process_resolution(res, cfg, device)
        frames.append(df)
        log.info("  %dpx: %d (image, class) pairs", res, len(df))

    combined = pd.concat(frames, ignore_index=True)

    # Merge to attach area + class_name
    merged = combined.merge(area_df, on=["image_idx", "class_idx"], how="left")

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cfg.output, index=False)
    log.info("Saved %s  (%d rows, %.1f MB)", cfg.output, len(merged), cfg.output.stat().st_size / 1e6)


if __name__ == "__main__":
    main(tyro.cli(Config))
