"""Run a CanViT episode: T glimpses sampled by a policy, recurrent state updated each step."""

from dataclasses import dataclass
from typing import Protocol

from canvit_pytorch import CanViTOutput, RecurrentState, Viewpoint, sample_at_viewpoint
from canvit_pytorch.patcher.foveated import FoveatedPatcher
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
    glimpse_px: int,
    state: RecurrentState | None = None,
) -> list[EpisodeStep]:
    B = images.shape[0]
    if state is None:
        state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)

    # Foveated patcher consumes the full image (foveates around viewpoint.centers
    # internally). Uniform patcher, as wrapped by the downstream classification /
    # segmentation models (glimpse_size_px=None), expects a pre-cropped glimpse.
    is_foveated = isinstance(getattr(model, "patcher", None), FoveatedPatcher)

    steps: list[EpisodeStep] = []
    for t in range(n_timesteps):
        vp = policy.step(t, state)
        if is_foveated:
            model_input = images
        else:
            model_input = sample_at_viewpoint(
                spatial=images, viewpoint=vp, glimpse_size_px=glimpse_px,
            )
        out = model(image=model_input, state=state, viewpoint=vp)
        state = out.state
        steps.append(EpisodeStep(t=t, state=state, output=out, viewpoint=vp))

    return steps
