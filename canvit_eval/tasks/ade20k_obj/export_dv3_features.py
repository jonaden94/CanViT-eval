"""Export DINOv3 patch features for ADE20K validation images to a single .pt."""

import logging
import time
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
from canvit_eval.tasks.ade20k_obj.dataframe_dataset import EXPECTED_N_VAL_IMAGES
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


def _sanity_check_saved(out_path: Path, expected_n_images: int, expected_grid: int, expected_embed_dim: int) -> None:
    """Reload the .pt we just wrote, assert shapes + key presence."""
    data = torch.load(out_path, map_location="cpu", weights_only=False)
    required = {"feats", "masks", "image_names", "grid", "embed_dim",
                "eval_resolution_px", "scene_size_px", "teacher_repo", "resize_mode"}
    missing = required - data.keys()
    assert not missing, f"reloaded .pt missing keys: {missing}"
    feats, masks = data["feats"], data["masks"]
    assert feats.shape == (expected_n_images, expected_grid * expected_grid, expected_embed_dim), feats.shape
    assert masks.shape[0] == expected_n_images, masks.shape
    assert data["grid"] == expected_grid, (data["grid"], expected_grid)
    assert data["embed_dim"] == expected_embed_dim, (data["embed_dim"], expected_embed_dim)
    log.info("reload sanity: feats=%s %s, masks=%s %s, keys=%d",
             feats.shape, feats.dtype, masks.shape, masks.dtype, len(data))


def main(cfg: ExportFeaturesConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    assert cfg.eval_resolution_px > 0, cfg.eval_resolution_px
    assert cfg.scene_size_px > 0, cfg.scene_size_px
    assert cfg.batch_size > 0, cfg.batch_size

    device = torch.device(cfg.device)
    out_path = features_path(cfg.eval_resolution_px)

    log.info("=== export DINOv3 features ===")
    log.info("teacher_repo=%s  device=%s  amp=%s", cfg.teacher_repo, device, cfg.amp)
    log.info("eval_resolution_px=%d  scene_size_px=%d  resize_mode=%s",
             cfg.eval_resolution_px, cfg.scene_size_px, cfg.resize_mode)
    log.info("ade20k_root=%s", cfg.ade20k_root)
    log.info("out_path=%s", out_path)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    teacher = load_teacher(cfg.teacher_repo, device)
    patch_size_px = teacher.model.config.patch_size
    embed_dim = teacher.embed_dim
    assert cfg.eval_resolution_px % patch_size_px == 0, (
        f"eval_resolution_px={cfg.eval_resolution_px} not divisible by teacher patch_size={patch_size_px}"
    )
    grid = cfg.eval_resolution_px // patch_size_px
    log.info("patch_size_px=%d  grid=%d  embed_dim=%d", patch_size_px, grid, embed_dim)

    img_tf, mask_tf = make_val_transforms(cfg.scene_size_px, cfg.resize_mode)
    dataset = ADE20kDataset(
        root=cfg.ade20k_root, split="validation",
        img_transform=img_tf, mask_transform=mask_tf,
    )
    n_images = len(dataset)
    assert n_images == EXPECTED_N_VAL_IMAGES, (
        f"ADE20K val should have {EXPECTED_N_VAL_IMAGES} images, loader reports {n_images}"
    )

    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
    feats_all = torch.empty(
        n_images, grid * grid, embed_dim, dtype=torch.float32, pin_memory=True,
    )
    masks_all = torch.empty(n_images, cfg.scene_size_px, cfg.scene_size_px, dtype=torch.uint8)
    image_names: list[str] = [""] * n_images

    feats_gb = n_images * feats_all[0].numel() * feats_all.element_size() / 1e9
    masks_mb = n_images * masks_all[0].numel() * masks_all.element_size() / 1e6
    log.info("amp_dtype=%s  n_images=%d  feats buffer=%.2f GB (%s)  masks buffer=%.1f MB (%s)",
             amp_dtype, n_images, feats_gb, feats_all.dtype, masks_mb, masks_all.dtype)

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    batch_start = 0
    t0 = time.monotonic()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp):
        for images, masks in tqdm(loader, desc="export", total=len(loader)):
            B = images.shape[0]
            assert images.shape[1:] == (3, cfg.scene_size_px, cfg.scene_size_px), images.shape
            assert masks.shape[-2:] == (cfg.scene_size_px, cfg.scene_size_px), masks.shape

            images = images.to(device, non_blocking=True)
            resized = F.interpolate(
                images, size=(cfg.eval_resolution_px, cfg.eval_resolution_px),
                mode="bilinear", align_corners=False,
            )
            feats = teacher.forward_norm_features(resized).patches
            assert feats.shape == (B, grid * grid, embed_dim), feats.shape

            # GPU → pinned-CPU copies with non_blocking=True are queued on the
            # CUDA stream; the pinned buffer is filled asynchronously by the
            # DMA engine. Keeps the export loop off the critical path.
            feats_all[batch_start : batch_start + B].copy_(feats.float(), non_blocking=True)
            masks_all[batch_start : batch_start + B].copy_(masks.to(torch.uint8), non_blocking=True)
            for i in range(B):
                image_names[batch_start + i] = dataset.images[batch_start + i].stem
            batch_start += B

    # Required for correctness: torch.save below reads feats_all/masks_all
    # directly, and those bytes are written by the async DMA queued above.
    # Without this sync, the save can race the copies and serialize garbage.
    if device.type == "cuda":
        torch.cuda.synchronize()

    assert batch_start == n_images, (batch_start, n_images)
    assert all(n != "" for n in image_names), "some image_names left blank"
    assert len(set(image_names)) == n_images, f"duplicate image_names: {n_images - len(set(image_names))}"
    elapsed = time.monotonic() - t0
    log.info("loop done in %.1fs (%.2f images/s)", elapsed, n_images / elapsed if elapsed > 0 else float("inf"))

    log.info("saving %s ...", out_path)
    torch.save({
        "feats": feats_all,
        "masks": masks_all,
        "image_names": image_names,
        "grid": grid,
        "embed_dim": embed_dim,
        "eval_resolution_px": cfg.eval_resolution_px,
        "scene_size_px": cfg.scene_size_px,
        "teacher_repo": cfg.teacher_repo,
        "resize_mode": cfg.resize_mode,
    }, out_path)
    size_gb = out_path.stat().st_size / 1e9
    log.info("saved %s  (%.2f GB)", out_path, size_gb)
    assert size_gb > 0.01, f"output file suspiciously small: {size_gb:.3f} GB"

    _sanity_check_saved(out_path, n_images, grid, embed_dim)


if __name__ == "__main__":
    main(tyro.cli(ExportFeaturesConfig))
