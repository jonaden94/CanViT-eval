"""Viewing policies for CanViT evaluation.

Convention: centers are (y, x) in [-1, 1]. Scales are in (0, 1].
y grows downward (row index), x grows rightward (column index).
Matches canvit.viewpoint.

All policies implement the Policy protocol from episode.py:
    step(t: int, state: RecurrentState) -> Viewpoint
"""

import logging
from collections.abc import Callable
from typing import Literal

import torch
from canvit import RecurrentState, Viewpoint
from canvit_utils.policies import coarse_to_fine_viewpoints, random_viewpoints
from torch import Tensor

log = logging.getLogger(__name__)

PolicyName = Literal[
    "coarse_to_fine",
    "fine_to_coarse",
    "random",
    "full_then_random",
    "entropy_coarse_to_fine",
    "constant_full_scene",
]

GetSpatialFn = Callable[[Tensor], Tensor]


# ── Static policies ──────────────────────────────────────────────────────────


class StaticPolicy:
    """Pre-generated viewpoints, ignoring canvas state."""

    def __init__(self, name: str, viewpoints: list[Viewpoint]) -> None:
        self.name = name
        self._viewpoints = viewpoints

    def step(self, t: int, state: RecurrentState) -> Viewpoint:
        return self._viewpoints[t]


# ── Fine-to-coarse ───────────────────────────────────────────────────────────


def _level_viewpoints(level: int) -> list[tuple[float, float, float]]:
    """(y, x, scale) for all crops at a C2F level. Level 0 = full scene."""
    n = 2**level
    scale = 1.0 / n
    return [
        ((2 * row + 1) * scale - 1.0, (2 * col + 1) * scale - 1.0, scale)
        for row in range(n) for col in range(n)
    ]


def fine_to_coarse_viewpoints(
    batch_size: int, device: torch.device, n_viewpoints: int,
) -> list[Viewpoint]:
    """Reversed C2F: finest scale first, then coarser."""
    levels: list[list[tuple[float, float, float]]] = []
    total = 0
    level = 0
    while total < n_viewpoints:
        lvl_vps = _level_viewpoints(level)
        levels.append(lvl_vps)
        total += len(lvl_vps)
        level += 1
    levels.reverse()

    result: list[Viewpoint] = []
    for level_vps in levels:
        level_t = torch.tensor(level_vps, device=device, dtype=torch.float32)
        n = len(level_vps)
        perms = (torch.zeros(batch_size, 1, dtype=torch.long, device=device) if n == 1
                 else torch.stack([torch.randperm(n, device=device) for _ in range(batch_size)]))
        for i in range(n):
            if len(result) >= n_viewpoints:
                return result
            idx = perms[:, i]
            result.append(Viewpoint(centers=level_t[idx, :2], scales=level_t[idx, 2]))
    return result[:n_viewpoints]


# ── Entropy-guided C2F ───────────────────────────────────────────────────────


def _build_tile_masks(
    crop_centers: list[tuple[float, float, float]], canvas_grid: int, device: torch.device,
) -> Tensor:
    """[n_tiles, G, G] boolean masks for tiles at a given C2F level."""
    G = canvas_grid
    assert G > 0 and (G & (G - 1)) == 0, f"canvas_grid must be a power of 2, got {G}"
    coords = torch.linspace(-1 + 1 / G, 1 - 1 / G, G, device=device)
    crops_t = torch.tensor(crop_centers, device=device)
    cy, cx, s = crops_t[:, 0], crops_t[:, 1], crops_t[:, 2]
    row_in = (coords.unsqueeze(0) - cy.unsqueeze(1)).abs() <= s.unsqueeze(1)
    col_in = (coords.unsqueeze(0) - cx.unsqueeze(1)).abs() <= s.unsqueeze(1)
    return row_in.unsqueeze(2) & col_in.unsqueeze(1)


class EntropyGuidedC2F:
    """C2F with within-level ordering by segmentation probe entropy.

    At each C2F level, visits crops in order of decreasing entropy
    (most uncertain first). Zero additional compute — the probe is already
    needed for mIoU anyway.
    """

    name = "entropy_coarse_to_fine"

    def __init__(
        self, batch_size: int, device: torch.device, canvas_grid: int,
        *, probe: torch.nn.Module, get_spatial_fn: GetSpatialFn,
    ) -> None:
        self._batch_size = batch_size
        self._device = device
        self._canvas_grid = canvas_grid
        self._probe = probe
        self._get_spatial_fn = get_spatial_fn

        self._levels = [_level_viewpoints(lvl) for lvl in range(3)]
        self._level_starts = []
        t = 0
        for lvl in self._levels:
            self._level_starts.append(t)
            t += len(lvl)

        self._tile_masks: list[Tensor | None] = [None]
        for lvl in range(1, 3):
            self._tile_masks.append(_build_tile_masks(self._levels[lvl], canvas_grid, device))

        self._visited: list[Tensor | None] = [None for _ in self._levels]

    def _compute_entropy(self, state: RecurrentState) -> Tensor:
        spatial = self._get_spatial_fn(state.canvas)
        B, G = spatial.shape[0], self._canvas_grid
        logits = self._probe(spatial.view(B, G, G, -1).float())
        log_probs = torch.log_softmax(logits, dim=1)
        return -(log_probs.exp() * log_probs).sum(dim=1)

    def step(self, t: int, state: RecurrentState) -> Viewpoint:
        level_idx = sum(1 for s in self._level_starts[1:] if t >= s)
        pos_in_level = t - self._level_starts[level_idx]
        crops = self._levels[level_idx]
        B = self._batch_size

        if level_idx == 0:
            cy, cx, s = crops[0]
            return Viewpoint(
                centers=torch.tensor([[cy, cx]], device=self._device).expand(B, -1),
                scales=torch.full((B,), s, device=self._device),
            )

        if pos_in_level == 0:
            self._visited[level_idx] = torch.zeros(B, len(crops), dtype=torch.bool, device=self._device)

        entropy = self._compute_entropy(state)
        masks = self._tile_masks[level_idx]
        assert masks is not None
        visited = self._visited[level_idx]
        assert visited is not None

        n_cells = masks.sum(dim=(1, 2)).clamp(min=1).float()
        scores = (entropy.unsqueeze(1) * masks.unsqueeze(0).float()).sum(dim=(2, 3)) / n_cells.unsqueeze(0)
        scores = scores.masked_fill(visited, float("-inf"))
        chosen = scores.argmax(dim=1)
        visited.scatter_(1, chosen.unsqueeze(1), True)

        all_crops = torch.tensor(crops, device=self._device)
        selected = all_crops[chosen]
        return Viewpoint(centers=selected[:, :2], scales=selected[:, 2])


# ── Factory ──────────────────────────────────────────────────────────────────


def make_policy(
    name: PolicyName,
    batch_size: int,
    device: torch.device,
    n_viewpoints: int,
    *,
    canvas_grid: int = 32,
    min_scale: float = 0.05,
    max_scale: float = 1.0,
    probe: torch.nn.Module | None = None,
    get_spatial_fn: GetSpatialFn | None = None,
) -> StaticPolicy | EntropyGuidedC2F:
    """Create a viewing policy by name. No aliases — use canonical names only."""
    if name == "coarse_to_fine":
        return StaticPolicy(name, coarse_to_fine_viewpoints(batch_size, device, n_viewpoints))
    if name == "fine_to_coarse":
        return StaticPolicy(name, fine_to_coarse_viewpoints(batch_size, device, n_viewpoints))
    if name == "full_then_random":
        return StaticPolicy(name, random_viewpoints(
            batch_size, device, n_viewpoints, min_scale=min_scale, max_scale=max_scale, start_with_full_scene=True,
        ))
    if name == "random":
        return StaticPolicy(name, random_viewpoints(
            batch_size, device, n_viewpoints, min_scale=min_scale, max_scale=max_scale, start_with_full_scene=False,
        ))
    if name == "constant_full_scene":
        return StaticPolicy(name, [Viewpoint.full_scene(batch_size=batch_size, device=device)] * n_viewpoints)
    if name == "entropy_coarse_to_fine":
        assert probe is not None and get_spatial_fn is not None, \
            "entropy_coarse_to_fine requires probe= and get_spatial_fn="
        return EntropyGuidedC2F(batch_size, device, canvas_grid, probe=probe, get_spatial_fn=get_spatial_fn)
    raise ValueError(f"Unknown policy: {name!r}")
