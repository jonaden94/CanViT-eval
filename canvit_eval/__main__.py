"""CanViT evaluation CLI.

    uv run python -m canvit_eval ade20k-seg-canvit --probe-repo canvit/probe-... [--episode.policy ...]
    uv run python -m canvit_eval ade20k-seg-dinov3 --probe-repo canvit/... --eval-resolution 128
    uv run python -m canvit_eval in1k-clf
    uv run python -m canvit_eval reconstruction --model-repo canvit/canvitb16-abl-...
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Union

import tyro

from canvit_eval.config import TEACHER_REPO, EpisodeConfig


@dataclass
class ADE20kSegCanViTCmd:
    """ADE20K segmentation with a CanViT backbone (multi-timestep episode)."""
    probe_repo: str
    output: Path = Path("results/ade20k_seg.pt")
    scene_size: int = 512
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)

    def run(self) -> None:
        from canvit_eval.tasks.ade20k_seg import CanViTConfig, run_canvit
        run_canvit(CanViTConfig(
            probe_repo=self.probe_repo, episode=self.episode,
            output=self.output, scene_size=self.scene_size,
            device=self.device, batch_size=self.batch_size, num_workers=self.num_workers,
        ))


@dataclass
class ADE20kSegDINOv3Cmd:
    """ADE20K segmentation with a DINOv3 backbone (single passive forward pass)."""
    probe_repo: str
    eval_resolution: int
    """Resolution the probe was trained at (e.g. 128, 192, 512). Teacher is run at this size,
    NOT at scene_size. Required — no default, since a mismatch silently degrades mIoU."""
    teacher_repo: str = TEACHER_REPO
    output: Path = Path("results/ade20k_seg.pt")
    scene_size: int = 512
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8

    def run(self) -> None:
        from canvit_eval.tasks.ade20k_seg import DINOv3Config, run_dinov3
        run_dinov3(DINOv3Config(
            probe_repo=self.probe_repo, teacher_repo=self.teacher_repo,
            eval_resolution=self.eval_resolution,
            output=self.output, scene_size=self.scene_size,
            device=self.device, batch_size=self.batch_size, num_workers=self.num_workers,
        ))


@dataclass
class IN1KClfCmd:
    """ImageNet-1K classification."""
    mode: Literal["finetuned", "frozen"] = "finetuned"
    episode: EpisodeConfig = field(default_factory=lambda: EpisodeConfig(canvas_grid=32))
    probe_repo: str = "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"
    val_dir: Path | None = None
    output: Path = Path("results/in1k_clf.pt")
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 8

    def run(self) -> None:
        from canvit_eval.config import imagenet_val_dir
        from canvit_eval.tasks.in1k_clf import Config, evaluate
        val = self.val_dir or imagenet_val_dir()
        evaluate(Config(
            mode=self.mode, episode=self.episode, probe_repo=self.probe_repo,
            val_dir=val, output=self.output,
            device=self.device, batch_size=self.batch_size, num_workers=self.num_workers,
        ))


@dataclass
class ReconCmd:
    """Reconstruction quality (cosine sim to DINOv3 teacher)."""
    model_repo: str
    episode: EpisodeConfig = field(default_factory=lambda: EpisodeConfig(policy="random", n_timesteps=10))
    output: Path = Path("results/reconstruction.pt")
    scene_size: int = 512
    device: str = "cuda"
    batch_size: int = 16
    teacher_cache: Path | None = None

    def run(self) -> None:
        from canvit_eval.tasks.reconstruction import Config, evaluate
        evaluate(Config(
            model_repo=self.model_repo, episode=self.episode,
            output=self.output, scene_size=self.scene_size,
            device=self.device, batch_size=self.batch_size,
            teacher_cache=self.teacher_cache,
        ))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = tyro.cli(Union[  # type: ignore[reportCallIssue]  # basedpyright can't resolve Union as TypeForm
        Annotated[ADE20kSegCanViTCmd, tyro.conf.subcommand("ade20k-seg-canvit")],
        Annotated[ADE20kSegDINOv3Cmd, tyro.conf.subcommand("ade20k-seg-dinov3")],
        Annotated[IN1KClfCmd, tyro.conf.subcommand("in1k-clf")],
        Annotated[ReconCmd, tyro.conf.subcommand("reconstruction")],
    ])
    cmd.run()


if __name__ == "__main__":
    main()
