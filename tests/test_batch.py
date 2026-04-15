"""Tests for batch eval matrix generation — pure, no GPU/data needed."""

import re
from pathlib import Path

from canvit_eval.batch import (
    ABLATION_REPOS,
    ADE20K_RESOLUTIONS,
    ALL_POLICIES,
    ALL_TASKS,
    CANVAS_GRIDS,
    DETERMINISTIC,
    DINOV3_RESOLUTIONS,
    DINOV3_VARIANTS,
    EXTRA_CANVAS_GRIDS,
    build_eval_matrix,
    filter_jobs,
)

_TS_RE = re.compile(r"\d{8}T\d{6}Z")

# Expected job counts derived from the matrix constants — recomputed here
# so the test fails loudly when the matrix changes, rather than silently.
_N_CANVIT_POLICY_N1 = len(ADE20K_RESOLUTIONS) * len(ALL_POLICIES)
_N_DINOV3 = len(DINOV3_VARIANTS) * len(DINOV3_RESOLUTIONS)
_N_CANVIT_T0 = len(CANVAS_GRIDS)
_N_ADE20K_N1 = _N_CANVIT_POLICY_N1 + _N_DINOV3 + _N_CANVIT_T0  # 12 + 14 + 4 = 30
_N_IN1K_N1_BOTH_MODES = 2 * 4  # 2 modes × 4 policies, all non-deterministic in IN1K_POLICIES
_N_RECON_N1 = len(ABLATION_REPOS)


def test_all_policies_from_literal():
    assert set(ALL_POLICIES) >= {"coarse_to_fine", "entropy_coarse_to_fine", "repeated_full_scene"}
    assert len(ALL_POLICIES) == 6


def test_deterministic_is_subset():
    assert DETERMINISTIC.issubset(set(ALL_POLICIES))


def test_entropy_c2f_is_deterministic():
    """Argmax over entropy scores is deterministic; batch trims to n=1."""
    assert "entropy_coarse_to_fine" in DETERMINISTIC


def test_ade20k_seg_n1():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    assert len(jobs) == _N_ADE20K_N1
    assert all(j.task == "ade20k-seg" for j in jobs)


def test_ade20k_seg_n5_trims_deterministic():
    """Non-deterministic policies get 5 runs; deterministic get 1."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=5, n_timesteps=21, tasks=["ade20k-seg"])
    non_det = len([p for p in ALL_POLICIES if p not in DETERMINISTIC])
    det = len([p for p in ALL_POLICIES if p in DETERMINISTIC])
    expected_canvit = len(ADE20K_RESOLUTIONS) * (non_det * 5 + det * 1)
    assert len(jobs) == expected_canvit + _N_DINOV3 + _N_CANVIT_T0


def test_in1k_clf_n1_emits_both_modes():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["in1k-clf"])
    assert len(jobs) == _N_IN1K_N1_BOTH_MODES
    modes = {j.model for j in jobs}
    assert modes == {"canvit-frozen", "canvit-finetuned"}


def test_recon_n1():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["recon"])
    assert len(jobs) == _N_RECON_N1
    assert all(j.task == "recon" for j in jobs)


def test_all_tasks_combined():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=ALL_TASKS)
    assert len(jobs) == _N_ADE20K_N1 + _N_IN1K_N1_BOTH_MODES + _N_RECON_N1


def test_include_extra_grids_adds_jobs():
    """--include-extra-grids adds jobs for each EXTRA_CANVAS_GRIDS entry, modulo
    the entropy_coarse_to_fine policy which is skipped on non-power-of-2 grids."""
    base = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    extended = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21,
                                 tasks=["ade20k-seg"], include_extra_grids=True)
    added = len(extended) - len(base)
    # Per extra grid: ALL_POLICIES policy-curve jobs (- 1 if grid not power-of-2)
    # + 1 t=0 job. EXTRA_CANVAS_GRIDS currently holds c9/10/12/24 — all non-power-of-2.
    non_pow2 = sum(1 for _, g in EXTRA_CANVAS_GRIDS if (g & (g - 1)) != 0)
    expected_per_grid = len(ALL_POLICIES) + 1  # policy-curve + t=0
    assert added == len(EXTRA_CANVAS_GRIDS) * expected_per_grid - non_pow2


def test_entropy_c2f_skipped_on_non_power_of_two():
    """entropy_coarse_to_fine can't run on c9/10/12/24; builder must exclude them."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21,
                             tasks=["ade20k-seg"], include_extra_grids=True)
    offending = [j for j in jobs if j.policy == "entropy_coarse_to_fine"
                 and j.canvas_grid in {9, 10, 12, 24}]
    assert offending == [], f"entropy_c2f jobs leaked into non-power-of-2 grids: {offending}"


def test_output_paths_unique():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=5, n_timesteps=21, tasks=ALL_TASKS)
    paths = [j.output for j in jobs]
    assert len(paths) == len(set(paths)), "Duplicate output paths"


def test_all_outputs_under_task_subdirs():
    out_dir = Path("/tmp/test_results")
    jobs = build_eval_matrix(out_dir, n_runs=1, n_timesteps=21, tasks=ALL_TASKS)
    expected = {
        out_dir / "ade20k_seg",
        out_dir / "in1k_clf_frozen", out_dir / "in1k_clf_finetuned",
        out_dir / "recon",
    }
    assert {j.output.parent for j in jobs} == expected


def test_filenames_contain_timestamp():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=ALL_TASKS)
    for j in jobs:
        assert _TS_RE.search(j.output.name), f"No timestamp in {j.output.name}"


def test_jobs_carry_structural_metadata():
    """Every CanViT policy-curve job has scene/grid/policy; every DINOv3 job has input_px/canvas_grid."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    canvit_policy = [j for j in jobs if j.model == "canvit" and j.policy != "coarse_to_fine" or
                     (j.policy == "coarse_to_fine" and "_s" in j.output.stem and "_r" in j.output.stem)]
    for j in canvit_policy:
        assert j.scene_size is not None and j.canvas_grid is not None and j.policy is not None
    dinov3 = [j for j in jobs if j.model.startswith("dinov3-")]
    for j in dinov3:
        assert j.input_px is not None and j.canvas_grid is not None and j.canvas_grid == j.input_px // 16


def test_filter_by_policy():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    kept = filter_jobs(jobs, policies=["coarse_to_fine"])
    # Should drop DINOv3 (policy=None) and policies != coarse_to_fine; keep canvit c2f policy + t=0 jobs.
    assert all(j.policy == "coarse_to_fine" for j in kept)
    assert len(kept) == len(ADE20K_RESOLUTIONS) + len(CANVAS_GRIDS)  # 2 policy-curve + 4 t=0


def test_filter_by_grid():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    kept = filter_jobs(jobs, grids=[32])
    assert all(j.canvas_grid == 32 for j in kept)
    # 6 policies @ c32 + 1 t=0 @ c32 + DINOv3 at 512px (grid = 32)
    assert len(kept) == len(ALL_POLICIES) + 1 + len(DINOV3_VARIANTS)


def test_filter_combined():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    kept = filter_jobs(jobs, policies=["coarse_to_fine"], grids=[32])
    assert len(kept) == 2  # 1 policy-curve + 1 t=0


def test_skip_existing_is_timestamp_agnostic(tmp_path):
    """already_done() globs on the structural stem — matches any prior-run timestamp."""
    out_dir = tmp_path / "results"
    ade_dir = out_dir / "ade20k_seg"
    ade_dir.mkdir(parents=True)

    # Put a fake old file with a DIFFERENT timestamp than what build_eval_matrix generates now.
    (ade_dir / "coarse_to_fine_s512_c32_20260101T000000Z_r0.pt").touch()

    jobs = build_eval_matrix(out_dir, n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    target = next(j for j in jobs if j.policy == "coarse_to_fine"
                  and j.scene_size == 512 and j.canvas_grid == 32 and j.run_idx == 0
                  and j.input_px is None)
    assert target.already_done(), "should match timestamp-stripped glob of existing .pt"

    # A different policy's job should not be affected.
    other = next(j for j in jobs if j.policy == "fine_to_coarse"
                 and j.scene_size == 512 and j.canvas_grid == 32 and j.run_idx == 0)
    assert not other.already_done()
