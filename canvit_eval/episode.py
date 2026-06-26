"""Run a CanViT episode: T glimpses sampled by a policy, recurrent state updated each step."""

from dataclasses import dataclass, replace
from typing import Protocol

import torch
from canvit_pytorch import CanViTOutput, RecurrentState, Viewpoint, sample_at_viewpoint
from canvit_pytorch.patcher.foveated import FoveatedPatcher
from canvit_pytorch.patcher.square import SquarePatcher
from torch import Tensor


class CanViTModel(Protocol):
    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState: ...
    def __call__(self, *, image: Tensor, state: RecurrentState, viewpoint: Viewpoint) -> CanViTOutput: ...


class Policy(Protocol):
    def step(self, t: int, state: RecurrentState) -> Viewpoint: ...


@dataclass(frozen=True)
class EpisodeStep:
    t: int
    state: RecurrentState
    output: CanViTOutput
    viewpoint: Viewpoint


def run_episode(
    *,
    model: CanViTModel,
    images: Tensor,
    policy: Policy,
    n_timesteps: int,
    canvas_grid: int,
    glimpse_px: int | None = None,
    override_scale: float | None = None,
    state: RecurrentState | None = None,
) -> list[EpisodeStep]:
    """Run a T-step CanViT episode.

    ``override_scale`` is a patcher- and policy-agnostic view-scale override:
      * ``None`` (default): pass the policy's own scales through (e.g.
        coarse-to-fine actually zooms full → 0.5 → 0.25).
      * a float: override every glimpse's scale to this constant, keeping the
        policy's *centers*. For the uniform patcher this fixes the pre-crop
        zoom; for the foveated/square patchers (which honor ``viewpoint.scales``,
        ``fix_size = scale * H``) it fixes the sensor window (``1.0`` = full
        image). Use a fixed value to eval a fixed-scale / per-rollout model
        in-distribution, ``None`` to let a per-glimpse model follow the policy."""
    B = images.shape[0]
    if state is None:
        state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    # Foveated AND square patchers consume the full image and foveate / sample
    # internally around viewpoint.centers, honoring viewpoint.scales
    # (fix_size = scale * H) -- this is exactly how they are driven during
    # pretraining, where the model always receives the full image. Only the
    # uniform patcher, as wrapped by the downstream
    # classification / segmentation models (glimpse_size_px=None), expects a
    # pre-cropped glimpse. Routing the square patcher through the uniform
    # pre-crop path double-crops it (pre-cropped glimpse, then re-foveated),
    # which silently corrupts its samples and degrades with each finer glimpse.
    consumes_full_image = isinstance(
        getattr(model, "patcher", None), (FoveatedPatcher, SquarePatcher)
    )

    # The uniform patcher needs a pre-cropped glimpse whose pixel size matches
    # what the model trained on. Training used glimpse_size_px =
    # (glimpse_grid_size - 1) * patch_stride_px + patch_size_px (CanViT-pretrain
    # train/model.py); the patch-embed conv then yields exactly glimpse_grid_size
    # tokens/side. Cropping at any other size silently changes the per-glimpse
    # token count. Derive it from the model so it tracks the backbone's patch
    # size AND stride for ANY config (stride == patch reduces to grid × patch),
    # and HARD-GUARD against a token-count mismatch via the conv-output formula.
    if not consumes_full_image:
        patch_size = model.backbone.patch_size_px
        stride = getattr(model.backbone, "patch_stride_px", patch_size)
        glimpse_grid = getattr(model, "glimpse_grid_size", None)
        grid = glimpse_grid if glimpse_grid is not None else 8
        if glimpse_px is None:
            glimpse_px = (grid - 1) * stride + patch_size
        assert (glimpse_px - patch_size) % stride == 0 and glimpse_px >= patch_size, (
            f"glimpse_px={glimpse_px} incompatible with patch_size_px={patch_size}, "
            f"patch_stride_px={stride} (need (glimpse_px - patch) divisible by stride)"
        )
        tokens = (glimpse_px - patch_size) // stride + 1
        if glimpse_grid is not None:
            assert tokens == glimpse_grid, (
                f"glimpse_px={glimpse_px} → {tokens} tokens/side, but the model was trained "
                f"with glimpse_grid_size={glimpse_grid} tokens/side. The uniform patcher would "
                f"see a different token count than in training. Set episode.glimpse_px="
                f"{(glimpse_grid - 1) * stride + patch_size} (= (grid-1)·stride + patch), or "
                f"leave it None to derive automatically."
            )

    steps: list[EpisodeStep] = []
    for t in range(n_timesteps):
        vp = policy.step(t, state)
        # Patcher-agnostic scale override: pin every glimpse to a fixed zoom
        # while keeping the policy's centers (None -> policy's own scales).
        # Drives the uniform pre-crop and the foveated/square sensor window
        # (fix_size = scale * H) alike.
        if override_scale is not None:
            vp = replace(vp, scales=torch.full_like(vp.scales, float(override_scale)))
        if consumes_full_image:
            model_input = images
        else:
            model_input = sample_at_viewpoint(
                spatial=images, viewpoint=vp, glimpse_size_px=glimpse_px,
            )
        out = model(image=model_input, state=state, viewpoint=vp)
        state = out.state
        steps.append(EpisodeStep(t=t, state=state, output=out, viewpoint=vp))

    return steps
