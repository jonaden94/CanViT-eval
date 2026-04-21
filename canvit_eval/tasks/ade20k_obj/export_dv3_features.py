"""Export DINOv3 patch features for ADE20K validation images to a single .pt."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from canvit_pytorch.teacher import load_teacher
from canvit_specialize.datasets.ade20k import ADE20kDataset, ResizeMode, make_val_transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_eval.config import TEACHER_REPO, ade20k_root
from canvit_eval.tasks.ade20k_obj.paths import FEATURES_DIR, features_path

log = logging.getLogger(__name__)


@dataclass
class ExportFeaturesConfig:
    teacher_repo: str = TEACHER_REPO
    ade20k_root: Path = field(default_factory=ade20k_root)
    eval_resolution_px: int = 128
    """Resolution images are resized to before the teacher forward pass."""
    scene_size_px: int = 512
    """Val-transform target; masks are saved at this resolution."""
    resize_mode: ResizeMode = "squish"
    batch_size: int = 32
    num_workers: int = 8
    device: str = "cuda"
    amp: bool = True
    out_dir: Path = FEATURES_DIR


def main(cfg: ExportFeaturesConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    device = torch.device(cfg.device)
    out_path = features_path(cfg.eval_resolution_px)

    log.info("teacher_repo=%s  device=%s  amp=%s", cfg.teacher_repo, device, cfg.amp)
    log.info("eval_resolution_px=%d  scene_size_px=%d  resize_mode=%s",
             cfg.eval_resolution_px, cfg.scene_size_px, cfg.resize_mode)
    log.info("ade20k_root=%s", cfg.ade20k_root)
    log.info("out_path=%s", out_path)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    teacher = load_teacher(cfg.teacher_repo, device)
    patch_size_px = teacher.model.config.patch_size
    assert cfg.eval_resolution_px % patch_size_px == 0, (
        f"eval_resolution_px={cfg.eval_resolution_px} not divisible by teacher patch_size={patch_size_px}"
    )
    grid = cfg.eval_resolution_px // patch_size_px
    embed_dim = teacher.embed_dim

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, cfg.resize_mode)
    dataset = ADE20kDataset(
        root=cfg.ade20k_root, split="validation",
        img_transform=img_tf, mask_transform=mask_tf,
    )
    n_images = len(dataset)
    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32

    feats_all = torch.empty(
        n_images, grid * grid, embed_dim, dtype=torch.float32, pin_memory=True,
    )
    masks_all = torch.empty(n_images, cfg.scene_size_px, cfg.scene_size_px, dtype=torch.uint8)
    image_names: list[str] = [""] * n_images

    feats_gb = n_images * feats_all[0].numel() * feats_all.element_size() / 1e9
    masks_mb = n_images * masks_all[0].numel() * masks_all.element_size() / 1e6
    log.info("n_images=%d  grid=%d  embed_dim=%d  amp_dtype=%s",
             n_images, grid, embed_dim, amp_dtype)
    log.info("feats buffer: %.1f GB (float32)   masks buffer: %.1f MB (uint8)",
             feats_gb, masks_mb)

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    batch_start = 0
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp):
        for images, masks in tqdm(loader, desc="export", total=len(loader)):
            B = images.shape[0]
            images = images.to(device, non_blocking=True)
            resized = F.interpolate(
                images, size=(cfg.eval_resolution_px, cfg.eval_resolution_px),
                mode="bilinear", align_corners=False,
            )
            feats = teacher.forward_norm_features(resized).patches
            assert feats.shape == (B, grid * grid, embed_dim), feats.shape

            feats_all[batch_start : batch_start + B].copy_(feats.float())
            masks_all[batch_start : batch_start + B].copy_(masks.to(torch.uint8))
            for i in range(B):
                image_names[batch_start + i] = dataset.images[batch_start + i].stem
            batch_start += B

    log.info("saving %s ...", out_path)
    torch.save({
        "feats": feats_all,
        "masks": masks_all,
        "image_names": image_names,
        "grid": grid,
        "embed_dim": embed_dim,
        "eval_resolution_px": cfg.eval_resolution_px,
        "scene_size_px": cfg.scene_size_px,
    }, out_path)
    log.info("saved %s  (%.1f GB)", out_path, out_path.stat().st_size / 1e9)


if __name__ == "__main__":
    main(tyro.cli(ExportFeaturesConfig))
