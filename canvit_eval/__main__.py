"""Unified CanViT evaluation CLI.

    uv run python -m canvit_eval ade20k-seg --probe-repo canvit/probe-...
    uv run python -m canvit_eval in1k-clf
    uv run python -m canvit_eval reconstruction --model-repo canvit/canvitb16-abl-...
"""

from typing import Annotated

import tyro

from canvit_eval.tasks.ade20k_seg import Config as ADE20kSegConfig, evaluate as eval_ade20k_seg
from canvit_eval.tasks.in1k_clf import Config as IN1KClfConfig, evaluate as eval_in1k_clf
from canvit_eval.tasks.reconstruction import Config as ReconConfig, evaluate as eval_recon


def main() -> None:
    cmd = tyro.cli(
        Annotated[ADE20kSegConfig, tyro.conf.subcommand("ade20k-seg")]
        | Annotated[IN1KClfConfig, tyro.conf.subcommand("in1k-clf")]
        | Annotated[ReconConfig, tyro.conf.subcommand("reconstruction")]
    )
    match cmd:
        case ADE20kSegConfig():
            eval_ade20k_seg(cmd)
        case IN1KClfConfig():
            eval_in1k_clf(cmd)
        case ReconConfig():
            eval_recon(cmd)


if __name__ == "__main__":
    main()
