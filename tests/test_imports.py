"""Structural smoke tests — lightweight invariants not covered by behavioral tests.

Kept minimal: only tests that catch regressions basedpyright/ruff can't. Pure
import checks are dropped — `uv run pytest` already exercises all imports
transitively, and basedpyright catches broken imports before runtime.
"""

from typing import get_args

import pytest
import torch

from canvit_eval.policies import PolicyName, make_policy


def test_static_policies_produce_valid_viewpoints():
    """Each non-entropy policy constructs and emits a viewpoint with correct shape + scale range."""
    static_names = [n for n in get_args(PolicyName) if n != "entropy_coarse_to_fine"]
    for name in static_names:
        policy = make_policy(name, batch_size=2, device=torch.device("cpu"), n_viewpoints=5)
        vp = policy.step(t=0, state=None)  # type: ignore[arg-type]  # StaticPolicy ignores state
        assert vp.centers.shape == (2, 2)
        assert vp.scales.shape == (2,)
        assert (vp.scales > 0).all()
        assert (vp.scales <= 1).all()


def test_entropy_policy_requires_probe_and_get_spatial():
    with pytest.raises(AssertionError):
        make_policy("entropy_coarse_to_fine", batch_size=1,
                    device=torch.device("cpu"), n_viewpoints=5)


def test_batch_size_by_scene_invariant():
    """Larger scene → smaller BS (OOM-safe). Verified via _BATCH_SIZE_BY_SCENE keys
    present in the matrix constants."""
    from canvit_eval.batch import _BATCH_SIZE_BY_SCENE
    assert _BATCH_SIZE_BY_SCENE[1024] < _BATCH_SIZE_BY_SCENE[512]


def test_every_cli_config_has_a_run_method():
    """The tyro dispatch in __main__ calls cfg.run() on every subcommand. Each
    Config must expose a no-arg run() method returning Path. Parametrized so an
    incomplete migration (adding a new Config class without run()) fails loudly."""
    from canvit_eval.tasks.ade20k_seg import CanViTConfig, DINOv3Config
    from canvit_eval.tasks.in1k_clf import Config as IN1KConfig
    from canvit_eval.tasks.reconstruction import Config as ReconConfig

    for cls in [CanViTConfig, DINOv3Config, IN1KConfig, ReconConfig]:
        assert callable(getattr(cls, "run", None)), \
            f"{cls.__name__} is missing a run() method (required by __main__ dispatch)"
