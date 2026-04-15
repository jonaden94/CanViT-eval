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
