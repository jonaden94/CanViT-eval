"""CanViT evaluation CLI — thin dispatch over the task Configs.

    uv run python -m canvit_eval ade20k-seg-canvit --probe-repo <org>/probe-... [--episode.policy ...]
    uv run python -m canvit_eval ade20k-seg-dinov3 --probe-repo <org>/... --eval-resolution 128
    uv run python -m canvit_eval in1k-clf
    uv run python -m canvit_eval reconstruction --model-repo <org>/canvitb16-abl-...

Each subcommand IS its task's Config dataclass (each has a `run()` method);
there's no separate CLI-level wrapper.
"""

import logging
from typing import Annotated, Union

import tyro

from canvit_eval.tasks.ade20k_seg import CanViTConfig, DINOv3Config
from canvit_eval.tasks.in1k_clf import Config as IN1KClfConfig
from canvit_eval.tasks.reconstruction import Config as ReconConfig


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = tyro.cli(Union[  # type: ignore[reportCallIssue]  # basedpyright can't resolve Union as TypeForm
        Annotated[CanViTConfig, tyro.conf.subcommand("ade20k-seg-canvit")],
        Annotated[DINOv3Config, tyro.conf.subcommand("ade20k-seg-dinov3")],
        Annotated[IN1KClfConfig, tyro.conf.subcommand("in1k-clf")],
        Annotated[ReconConfig, tyro.conf.subcommand("reconstruction")],
    ])
    cmd.run()


if __name__ == "__main__":
    main()
