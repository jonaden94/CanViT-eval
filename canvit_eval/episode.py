"""Run a CanViT episode: T glimpses sampled by a policy, recurrent state updated each step."""

from dataclasses import dataclass
from typing import Protocol

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
    state: RecurrentState | None = None,
) -> list[EpisodeStep]:
    B = images.shape[0]
    if state is None:
        state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    # Foveated AND square patchers consume the full image and foveate / sample
    # internally around viewpoint.centers (scale ignored) -- this is exactly how
    # they are driven during pretraining, where the model always receives the
    # full image. Only the uniform patcher, as wrapped by the downstream
    # classification / segmentation models (glimpse_size_px=None), expects a
    # pre-cropped glimpse. Routing the square patcher through the uniform
    # pre-crop path double-crops it (pre-cropped glimpse, then re-foveated),
    # which silently corrupts its samples and degrades with each finer glimpse.
    consumes_full_image = isinstance(
        getattr(model, "patcher", None), (FoveatedPatcher, SquarePatcher)
    )

    # The uniform patcher needs a pre-cropped glimpse whose pixel size matches
    # what the model trained on. Training used glimpse_size_px = glimpse_grid_size
    # × patch_size_px (CanViT-pretrain train/model.py); cropping at any other size
    # silently changes the per-glimpse token count (e.g. an 8px-patch model
    # trained at 64px would receive 256 tokens instead of 64 if cropped at 128px).
    # Derive it from the model so it tracks the backbone's patch size for ANY
    # patch size, and HARD-GUARD against a token-count mismatch.
    if not consumes_full_image:
        patch_size = model.backbone.patch_size_px
        glimpse_grid = getattr(model, "glimpse_grid_size", None)
        if glimpse_px is None:
            glimpse_px = (glimpse_grid or 8) * patch_size
        assert glimpse_px % patch_size == 0, (
            f"glimpse_px={glimpse_px} is not divisible by patch_size_px={patch_size}"
        )
        if glimpse_grid is not None:
            assert glimpse_px // patch_size == glimpse_grid, (
                f"glimpse_px={glimpse_px} → {glimpse_px // patch_size} tokens/side, but the "
                f"model was trained with glimpse_grid_size={glimpse_grid} tokens/side. The "
                f"uniform patcher would see a different token count than in training. "
                f"Set episode.glimpse_px={glimpse_grid * patch_size} (= glimpse_grid_size × "
                f"patch_size_px), or leave it None to derive automatically."
            )

    steps: list[EpisodeStep] = []
    for t in range(n_timesteps):
        vp = policy.step(t, state)
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
