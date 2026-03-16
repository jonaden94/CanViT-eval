"""ImageNet-1K classification: frozen CanViT CLS → destandardize → probe → top-1."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader

from canvit_eval.config import EpisodeConfig, imagenet_val_dir
from canvit_eval.datasets.imagenet import make_in1k_dataset
from canvit_eval.runner import eval_batches, load_model
from canvit_eval.utils import collect_metadata

log = logging.getLogger(__name__)
TOP_K = 5


def _load_probe(repo: str, device: torch.device) -> nn.Linear:
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    probe = nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0])
    probe.load_state_dict({"weight": sd["weight"], "bias": sd["bias"]})
    return probe.to(device).eval()


@dataclass
class Config:
    episode: EpisodeConfig = field(default_factory=lambda: EpisodeConfig(canvas_grid=32))
    probe_repo: str = "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"
    val_dir: Path = field(default_factory=imagenet_val_dir)
    output: Path = Path("results/in1k_clf.pt")
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 8
    amp: bool = True


@torch.inference_mode()
def evaluate(cfg: Config) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(cfg.device)

    model = load_model(cfg.episode.model_repo, device)
    probe = _load_probe(cfg.probe_repo, device)
    canvas_grid = cfg.episode.canvas_grid
    assert canvas_grid is not None, "canvas_grid required for IN1K"
    cls_std, _ = model.standardizers(canvas_grid)
    assert cls_std.initialized

    img_size = canvas_grid * model.backbone.patch_size_px
    dataset = make_in1k_dataset(cfg.val_dir, img_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    N, T = len(dataset), cfg.episode.n_timesteps

    all_top_k = torch.zeros(N, T, TOP_K, dtype=torch.int16)
    all_labels = torch.zeros(N, dtype=torch.int16)
    correct_top1 = torch.zeros(T, device=device, dtype=torch.long)
    t_start = time.monotonic()
    processed = 0

    for br in eval_batches(model=model, loader=loader, episode_cfg=cfg.episode,
                           canvas_grid=canvas_grid, device=device, amp=cfg.amp):
        _, labels = br.batch
        labels_gpu = labels.to(device, non_blocking=True)
        B = labels.shape[0]
        all_labels[processed:processed + B] = labels.to(torch.int16)

        for step in br.steps:
            cls_destd = cls_std.destandardize(model.predict_scene_teacher_cls(step.state.recurrent_cls))
            top_k = probe(cls_destd).topk(TOP_K, dim=-1).indices
            all_top_k[processed:processed + B, step.t] = top_k.cpu().to(torch.int16)
            correct_top1[step.t] += (top_k[:, 0] == labels_gpu).sum()
        processed += B

    for t in range(T):
        log.info("  t%d: top1=%.2f%%", t, 100 * correct_top1[t].item() / N)

    results = {
        "top_k_preds": all_top_k, "labels": all_labels,
        "metadata": {**collect_metadata(cfg), "wall_time_seconds": time.monotonic() - t_start, "n_images": N},
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, cfg.output)
    log.info("Saved to %s", cfg.output)
    return cfg.output
