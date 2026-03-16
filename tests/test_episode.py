"""Tests for the core episode runner.

Uses the real CanViT model from HuggingFace (tiny forward pass, CPU).
This is an integration smoke test, not a unit test.
"""

import torch
from canvit import Viewpoint
from canvit.model.pretraining.hub import CanViTForPretrainingHFHub

from canvit_eval.episode import EpisodeStep, run_episode


class _FullScenePolicy:
    """Always return full-scene viewpoint."""
    def step(self, t: int, state: object) -> Viewpoint:
        return Viewpoint.full_scene(batch_size=1, device=torch.device("cpu"))


def test_run_episode_shapes() -> None:
    """Smoke test: load real model, run 2 timesteps, check output shapes."""
    model = CanViTForPretrainingHFHub.from_pretrained(
        "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
    ).eval()
    images = torch.randn(1, 3, 512, 512)
    steps = run_episode(
        model=model, images=images, policy=_FullScenePolicy(),
        n_timesteps=2, canvas_grid=8, glimpse_px=128,
    )
    assert len(steps) == 2
    assert all(isinstance(s, EpisodeStep) for s in steps)
    # Canvas shape: [B, n_regs + grid², canvas_dim]
    canvas = steps[-1].state.canvas
    assert canvas.shape[0] == 1
    assert canvas.shape[1] == 16 + 8 * 8  # 16 regs + 64 spatial
    assert canvas.shape[2] == 1024  # canvas_dim


def test_episode_canvas_evolves() -> None:
    """Canvas should change between timesteps (not frozen)."""
    model = CanViTForPretrainingHFHub.from_pretrained(
        "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
    ).eval()
    images = torch.randn(1, 3, 256, 256)
    steps = run_episode(
        model=model, images=images, policy=_FullScenePolicy(),
        n_timesteps=2, canvas_grid=8, glimpse_px=128,
    )
    c0 = steps[0].state.canvas
    c1 = steps[1].state.canvas
    assert not torch.equal(c0, c1), "Canvas should change between timesteps"
