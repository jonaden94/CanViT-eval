"""Tests for batch eval matrix generation — pure, no GPU/data needed."""

import re
from pathlib import Path

from canvit_eval.batch import ALL_POLICIES, DETERMINISTIC, build_eval_matrix

_TS_RE = re.compile(r"\d{8}T\d{6}Z")


def test_all_policies_from_literal():
    """ALL_POLICIES derived from PolicyName Literal, not hardcoded."""
    assert "coarse_to_fine" in ALL_POLICIES
    assert "entropy_coarse_to_fine" in ALL_POLICIES
    assert "constant_full_scene" in ALL_POLICIES
    assert len(ALL_POLICIES) == 6


def test_deterministic_subset():
    assert DETERMINISTIC.issubset(set(ALL_POLICIES))


def test_build_eval_matrix_n1():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21)
    # 6 policies × 2 res × 1 run = 12 CanViT multi-T
    # + 14 DINOv3 (7 ViT-B + 7 ViT-S)
    # + 4 single-glimpse probes
    assert len(jobs) == 30


def test_build_eval_matrix_n5():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=5, n_timesteps=21)
    # 5 stochastic policies × 2 res × 5 runs = 50
    # + 1 deterministic × 2 res × 1 run = 2
    # + 14 DINOv3 + 4 single-glimpse = 18
    assert len(jobs) == 70  # 50 + 2 + 14 + 4


def test_output_paths_unique():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=5, n_timesteps=21)
    paths = [j.output for j in jobs]
    assert len(paths) == len(set(paths)), "Duplicate output paths!"


def test_all_outputs_under_out_dir():
    out_dir = Path("/tmp/test_results")
    jobs = build_eval_matrix(out_dir, n_runs=1, n_timesteps=21)
    for j in jobs:
        assert j.output.parent == out_dir


def test_filenames_contain_timestamp():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21)
    for j in jobs:
        assert _TS_RE.search(j.output.name), f"No timestamp in {j.output.name}"
