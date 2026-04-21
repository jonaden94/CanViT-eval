"""Export DINOv3 patch features for ADE20K validation images.

Loads every image in the ADE20K validation split, runs a DINOv3 teacher, and
writes a single .pt file to output/dv3_features/{eval_resolution}px_features.pt.

Saved dict:
    feats        – float32 [N, grid*grid, D]  (pinned, then saved)
    masks        – uint8   [N, scene_size, scene_size]  (IGNORE_LABEL=255)
    image_names  – list[str], length N
    grid         – int
    embed_dim    – int
    eval_resolution – int
    scene_size   – int
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from canvit_specialize.datasets.ade20k import ADE20kDataset, ResizeMode, make_val_transforms
from canvit_pytorch.teacher import load_teacher
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_eval.tasks.ade20k_obj.dataframe_dataset import ade20k_root

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")


ALL_RESOLUTIONS: list[int] = [128, 144, 160, 192, 256, 384, 512]


@dataclass
class Config:
    teacher_repo: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    probe_repo: str = ""
    ade20k_root: Path = field(default_factory=ade20k_root)
    eval_resolution: int = 512
    """Resolution images are resized to before the teacher forward pass. grid = eval_resolution // 16."""
    all: bool = False
    """Export features for all supported resolutions: 128, 144, 160, 192, 256, 384, 512."""
    scene_size: int = 512
    """Val-transform resize target; masks are saved at this resolution."""
    resize_mode: ResizeMode = "squish"
    batch_size: int = 32
    num_workers: int = 8
    device: str = "auto"
    amp: bool = True
    out_dir: Path = Path("output/dv3_features/")


def _get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(cfg: Config) -> None:
    device = _get_device(cfg.device)
    log.info("device=%s", device)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / f"{cfg.eval_resolution}px_features.pt"

    teacher = load_teacher(cfg.teacher_repo, device)
    D = teacher.embed_dim
    grid = cfg.eval_resolution // 16
    log.info("teacher_repo=%s  eval_resolution=%d  grid=%d  D=%d", cfg.teacher_repo, cfg.eval_resolution, grid, D)

    img_tf, mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    dataset = ADE20kDataset(
        root=cfg.ade20k_root, split="validation",
        img_transform=img_tf, mask_transform=mask_tf,
    )
    N = len(dataset)
    log.info("val set: %d images  =>  feats %.1f GB  masks %.1f MB",
             N,
             N * grid * grid * D * 4 / 1e9,
             N * cfg.scene_size * cfg.scene_size / 1e6)

    # Pinned CPU buffers — float32 feats allow non_blocking async DMA from GPU
    feats_all = torch.empty(N, grid * grid, D, dtype=torch.float32, pin_memory=True)
    masks_all = torch.empty(N, cfg.scene_size, cfg.scene_size, dtype=torch.uint8)
    image_names: list[str] = [""] * N

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
    batch_start = 0

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp):
        for images, masks in tqdm(loader, desc="export", total=len(loader)):
            B = images.shape[0]
            images = images.to(device, non_blocking=True)
            resized = F.interpolate(
                images, size=(cfg.eval_resolution, cfg.eval_resolution),
                mode="bilinear", align_corners=False,
            )
            feats = teacher.forward_norm_features(resized).patches  # [B, grid*grid, D], bfloat16

            # async DMA: GPU → pinned CPU, no sync stall
            feats_all[batch_start : batch_start + B].copy_(feats.float(), non_blocking=True)
            masks_all[batch_start : batch_start + B].copy_(masks.to(torch.uint8), non_blocking=True)
            for i in range(B):
                image_names[batch_start + i] = dataset.images[batch_start + i].stem
            batch_start += B

    # one sync after the loop, not per-batch
    if device.type == "cuda":
        torch.cuda.synchronize()

    log.info("saving %s ...", out_path)
    torch.save({
        "feats": feats_all,
        "masks": masks_all,
        "image_names": image_names,
        "grid": grid,
        "embed_dim": D,
        "eval_resolution": cfg.eval_resolution,
        "scene_size": cfg.scene_size,
    }, out_path)
    log.info("saved %s  (%.1f GB)", out_path, out_path.stat().st_size / 1e9)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    if cfg.all:
        for res in ALL_RESOLUTIONS:
            cfg.eval_resolution = res
            main(cfg)
    else:
        main(cfg)
