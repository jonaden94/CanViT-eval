"""Unified CanViT evaluation CLI.

    uv run python -m canvit_eval ade20k-seg --model canvit --probe-repo ...
    uv run python -m canvit_eval ade20k-seg --model dinov3 --probe-repo ...
    uv run python -m canvit_eval in1k-clf
    uv run python -m canvit_eval reconstruction --model-repo ...
"""

import logging
from dataclasses import dataclass, field
from typing import Annotated, Literal

import torch
import tyro
from canvit.model.pretraining.hub import CanViTForPretrainingHFHub
from canvit_utils.teacher import load_teacher

from canvit_eval.config import DEFAULT_MODEL_REPO, EpisodeConfig, HardwareConfig
from canvit_eval.features import canvit_extractor, dinov3_extractor
from canvit_eval.tasks.ade20k_seg import Config as ADE20kConfig, evaluate as eval_ade20k
from canvit_eval.tasks.in1k_clf import Config as IN1KConfig, evaluate as eval_in1k
from canvit_eval.tasks.reconstruction import Config as ReconConfig, evaluate as eval_recon

log = logging.getLogger(__name__)

TEACHER_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"


@dataclass
class ADE20kSegCmd:
    """ADE20K segmentation with CanViT or DINOv3."""

    model: Literal["canvit", "dinov3"] = "canvit"
    cfg: ADE20kConfig = field(default_factory=lambda: ADE20kConfig(probe_repo="REQUIRED"))
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    hw: HardwareConfig = field(default_factory=HardwareConfig)

    def run(self) -> None:
        device = torch.device(self.cfg.device)

        if self.model == "canvit":
            model = CanViTForPretrainingHFHub.from_pretrained(self.episode.model_repo).to(device).eval()
            canvas_grid = self.episode.canvas_grid or self.cfg.scene_size // model.backbone.patch_size_px
            extract = canvit_extractor(
                model, policy_name=self.episode.policy, n_timesteps=self.episode.n_timesteps,
                canvas_grid=canvas_grid, glimpse_px=self.episode.glimpse_px,
                min_scale=self.episode.min_scale, max_scale=self.episode.max_scale,
            )
            meta = {"model_repo": self.episode.model_repo, "canvas_grid": canvas_grid,
                    "policy": self.episode.policy, "n_timesteps": self.episode.n_timesteps}
        else:
            teacher = load_teacher(TEACHER_REPO, device)
            extract = dinov3_extractor(teacher, eval_resolution=self.cfg.scene_size)
            meta = {"model": "dinov3", "eval_resolution": self.cfg.scene_size}

        eval_ade20k(self.cfg, extract, metadata=meta)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = tyro.cli(
        Annotated[ADE20kSegCmd, tyro.conf.subcommand("ade20k-seg")]
        | Annotated[IN1KConfig, tyro.conf.subcommand("in1k-clf")]
        | Annotated[ReconConfig, tyro.conf.subcommand("reconstruction")]
    )
    match cmd:
        case ADE20kSegCmd():
            cmd.run()
        case IN1KConfig():
            eval_in1k(cmd)
        case ReconConfig():
            eval_recon(cmd)


if __name__ == "__main__":
    main()
