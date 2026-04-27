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
    EXTRA_IN1K_RESOLUTIONS,
    IN1K_POLICIES,
    IN1K_RESOLUTIONS,
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
_N_IN1K_POLICIES = len(IN1K_POLICIES)
_N_IN1K_N1_BOTH_MODES = len(IN1K_RESOLUTIONS) * _N_IN1K_POLICIES * 2  # baseline: frozen + finetuned
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
    """--include-extra-grids adds:
      - len(EXTRA_ADE20K_RESOLUTIONS) multi-step entries × len(ALL_POLICIES) policies
        (minus entropy_c2f on non-power-of-2 grids)
      - len(EXTRA_CANVAS_GRIDS) additional t=0 jobs (disjoint from CANVAS_GRIDS by design).
    """
    from canvit_eval.batch import EXTRA_ADE20K_RESOLUTIONS
    base = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    extended = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21,
                                 tasks=["ade20k-seg"], include_extra_grids=True)
    added = len(extended) - len(base)
    multi_non_pow2_skipped = sum(1 for _, g, _ in EXTRA_ADE20K_RESOLUTIONS if (g & (g - 1)) != 0)
    expected_multistep = len(EXTRA_ADE20K_RESOLUTIONS) * len(ALL_POLICIES) - multi_non_pow2_skipped
    expected_t0 = len(EXTRA_CANVAS_GRIDS)
    assert added == expected_multistep + expected_t0


def test_extra_canvas_grids_disjoint_from_canvas_grids():
    """Avoid regression: (512, 8) and (512, 16) must NOT be in EXTRA_CANVAS_GRIDS
    because they're already in CANVAS_GRIDS — duplicating them produces identical
    t=0 output paths."""
    assert set(EXTRA_CANVAS_GRIDS).isdisjoint(set(CANVAS_GRIDS))


def test_entropy_c2f_skipped_on_non_power_of_two():
    """entropy_coarse_to_fine can't run on non-power-of-2 grids; builder must exclude them.
    EXTRA_CANVAS_GRIDS may include mixed pow2 (c8, c16) and non-pow2 (c9, c10, c12, c24).
    The gate must apply only to the non-pow2 subset, not drop c8/c16."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21,
                             tasks=["ade20k-seg"], include_extra_grids=True)
    non_pow2_grids = {g for _, g in EXTRA_CANVAS_GRIDS if (g & (g - 1)) != 0}
    offending = [j for j in jobs if j.policy == "entropy_coarse_to_fine"
                 and j.canvas_grid in non_pow2_grids]
    assert offending == [], f"entropy_c2f jobs leaked into non-power-of-2 grids: {offending}"
    # And entropy_c2f SHOULD appear for power-of-2 extras (c8, c16) — sanity-check present.
    pow2_extra = {g for _, g in EXTRA_CANVAS_GRIDS if (g & (g - 1)) == 0}
    if pow2_extra:
        entropy_on_pow2 = [j for j in jobs if j.policy == "entropy_coarse_to_fine"
                           and j.canvas_grid in pow2_extra]
        assert entropy_on_pow2, "entropy_c2f should run on power-of-2 extra grids"


def test_output_paths_unique():
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=5, n_timesteps=21, tasks=ALL_TASKS)
    paths = [j.output for j in jobs]
    assert len(paths) == len(set(paths)), "Duplicate output paths"


def test_breadth_first_scheduling():
    """build_eval_matrix() emits jobs in breadth-first order: every r=0 runs
    before any r=1, etc. Round 0 covers every cell once, so an early
    interruption still yields n=1 per cell for a preview figure."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=5, n_timesteps=21, tasks=ALL_TASKS)
    positions_by_run_idx: dict[int, list[int]] = {}
    for i, j in enumerate(jobs):
        positions_by_run_idx.setdefault(j.run_idx, []).append(i)
    max_by_round = {r: max(ps) for r, ps in positions_by_run_idx.items()}
    min_by_round = {r: min(ps) for r, ps in positions_by_run_idx.items()}
    for r in sorted(positions_by_run_idx)[:-1]:
        assert max_by_round[r] < min_by_round[r + 1], (
            f"run_idx={r + 1} starts at position {min_by_round[r + 1]} "
            f"before run_idx={r} finishes at {max_by_round[r]} — not breadth-first"
        )


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


def test_skip_existing_is_per_run_idx(tmp_path):
    """Existing r=0 on disk must NOT satisfy r=1's already_done check —
    different run_idx values are independent samples."""
    out_dir = tmp_path / "results"
    ade_dir = out_dir / "ade20k_seg"
    ade_dir.mkdir(parents=True)

    # Only r=0 exists.
    (ade_dir / "coarse_to_fine_s512_c32_20260101T000000Z_r0.pt").touch()

    jobs = build_eval_matrix(out_dir, n_runs=2, n_timesteps=21, tasks=["ade20k-seg"])
    r0 = next(j for j in jobs if j.policy == "coarse_to_fine"
              and j.scene_size == 512 and j.canvas_grid == 32 and j.run_idx == 0
              and j.input_px is None)
    r1 = next(j for j in jobs if j.policy == "coarse_to_fine"
              and j.scene_size == 512 and j.canvas_grid == 32 and j.run_idx == 1
              and j.input_px is None)
    assert r0.already_done(), "existing r0 file should match r0 job"
    assert not r1.already_done(), \
        "r1 job must NOT match an existing r0 file (regression from 2026-04-14 bug)"


def test_skip_existing_respects_policy_boundary(tmp_path):
    """Existing coarse_to_fine r0 must NOT satisfy fine_to_coarse r0."""
    out_dir = tmp_path / "results"
    ade_dir = out_dir / "ade20k_seg"
    ade_dir.mkdir(parents=True)
    (ade_dir / "coarse_to_fine_s512_c32_20260101T000000Z_r0.pt").touch()

    jobs = build_eval_matrix(out_dir, n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    other = next(j for j in jobs if j.policy == "fine_to_coarse"
                 and j.canvas_grid == 32 and j.scene_size == 512)
    assert not other.already_done()


def test_skip_existing_respects_scene_and_grid(tmp_path):
    """Existing data at c32@s512 must NOT satisfy c64@s1024 or c9@s512."""
    out_dir = tmp_path / "results"
    ade_dir = out_dir / "ade20k_seg"
    ade_dir.mkdir(parents=True)
    (ade_dir / "coarse_to_fine_s512_c32_20260101T000000Z_r0.pt").touch()

    jobs = build_eval_matrix(out_dir, n_runs=1, n_timesteps=21,
                             tasks=["ade20k-seg"], include_extra_grids=True)
    c64 = next(j for j in jobs if j.policy == "coarse_to_fine"
               and j.canvas_grid == 64 and j.scene_size == 1024)
    c9 = next(j for j in jobs if j.policy == "coarse_to_fine"
              and j.canvas_grid == 9 and j.scene_size == 512)
    assert not c64.already_done()
    assert not c9.already_done()


def test_skip_existing_fills_partial_n_runs(tmp_path):
    """Integration-style: existing data at r=0..2, --n-runs 5 should leave r=0..2 in place
    and generate fresh r=3, r=4 jobs that are NOT marked done."""
    out_dir = tmp_path / "results"
    ade_dir = out_dir / "ade20k_seg"
    ade_dir.mkdir(parents=True)
    for run in range(3):
        (ade_dir / f"coarse_to_fine_s512_c32_20260101T000000Z_r{run}.pt").touch()

    jobs = build_eval_matrix(out_dir, n_runs=5, n_timesteps=21, tasks=["ade20k-seg"])
    c2f_c32 = [j for j in jobs if j.policy == "coarse_to_fine"
               and j.scene_size == 512 and j.canvas_grid == 32 and j.input_px is None]
    assert len(c2f_c32) == 5  # non-deterministic → full n_runs

    done = [j for j in c2f_c32 if j.already_done()]
    pending = [j for j in c2f_c32 if not j.already_done()]
    assert sorted(j.run_idx for j in done) == [0, 1, 2], \
        f"r0/r1/r2 should be done; got done={[j.run_idx for j in done]}"
    assert sorted(j.run_idx for j in pending) == [3, 4], \
        f"r3/r4 should be pending; got pending={[j.run_idx for j in pending]}"


def test_n_runs_for_stochastic_matches_n_runs():
    from canvit_eval.batch import _n_runs_for
    assert _n_runs_for("coarse_to_fine", 5) == 5
    assert _n_runs_for("fine_to_coarse", 10) == 10
    assert _n_runs_for("random", 1) == 1


def test_n_runs_for_deterministic_trimmed_to_one():
    from canvit_eval.batch import _n_runs_for
    assert _n_runs_for("entropy_coarse_to_fine", 5) == 1
    assert _n_runs_for("repeated_full_scene", 10) == 1
    assert _n_runs_for("entropy_coarse_to_fine", 1) == 1  # min of n_runs and 1


def test_dinov3_canvas_grid_derivation():
    """DINOv3 jobs must have canvas_grid = input_px // 16 (patch size)."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["ade20k-seg"])
    dv3 = [j for j in jobs if j.model.startswith("dinov3-")]
    assert dv3, "expected DINOv3 jobs in ade20k-seg batch"
    for j in dv3:
        assert j.input_px is not None and j.canvas_grid is not None
        assert j.canvas_grid == j.input_px // 16, \
            f"{j.output.name}: canvas_grid={j.canvas_grid} != input_px/16={j.input_px//16}"


def test_eval_job_output_path_uniqueness_at_matrix_build():
    """Within a single matrix build (i.e. single timestamp), no two jobs share an output path."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=10, n_timesteps=21, tasks=ALL_TASKS,
                             include_extra_grids=True)
    paths = [j.output for j in jobs]
    assert len(paths) == len(set(paths)), "duplicate output paths inside one build_eval_matrix call"


def test_eval_job_structural_tuple_is_unique():
    """Combined (task, model, policy, scene_size, canvas_grid, input_px, run_idx) must be unique —
    this is the skip-existing matching key."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=10, n_timesteps=21, tasks=ALL_TASKS,
                             include_extra_grids=True)
    keys = [(j.task, j.model, j.policy, j.scene_size, j.canvas_grid, j.input_px, j.run_idx)
            for j in jobs]
    assert len(keys) == len(set(keys)), "structural-identity tuple not unique across jobs"


# ── IN1k canvas-resolution sweep tests ──────────────────────────────────


def test_in1k_resolutions_disjoint():
    """Baseline vs extras must not overlap on (scene, grid) — otherwise
    include-extra-grids would duplicate baseline jobs."""
    baseline = {(s, g) for s, g, _ in IN1K_RESOLUTIONS}
    extras = {(s, g) for s, g, _ in EXTRA_IN1K_RESOLUTIONS}
    assert not (baseline & extras)


def test_in1k_filename_encodes_scene_and_grid():
    """IN1k outputs must carry s{scene}_c{grid} in the filename so the exporter
    can group runs by (policy, scene, grid)."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["in1k-clf"])
    in1k = [j for j in jobs if j.task == "in1k-clf"]
    assert in1k, "no in1k jobs"
    for j in in1k:
        assert j.scene_size is not None and j.canvas_grid is not None
        assert f"s{j.scene_size}_c{j.canvas_grid}" in j.output.name, (
            f"missing resolution fields in {j.output.name}"
        )


def test_in1k_jobs_carry_structural_fields():
    """in1k jobs must populate scene_size and canvas_grid (were None pre-refactor)."""
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["in1k-clf"])
    for j in jobs:
        if j.task == "in1k-clf":
            assert j.scene_size is not None
            assert j.canvas_grid is not None


def test_in1k_include_extras_sweeps_frozen_only():
    """Extra IN1k resolutions should expand frozen-mode jobs but NOT finetuned —
    the finetuned weights were specialized at s=512/c=32 so off-grid inference is
    a different (non-paper-claim) question."""
    base = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21, tasks=["in1k-clf"])
    ext = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21,
                            tasks=["in1k-clf"], include_extra_grids=True)

    base_frozen = [j for j in base if j.model == "canvit-frozen"]
    ext_frozen = [j for j in ext if j.model == "canvit-frozen"]
    base_ft = [j for j in base if j.model == "canvit-finetuned"]
    ext_ft = [j for j in ext if j.model == "canvit-finetuned"]

    # Frozen expands; finetuned is unchanged.
    assert len(ext_frozen) > len(base_frozen)
    assert len(ext_ft) == len(base_ft)

    # Extras cover exactly EXTRA_IN1K_RESOLUTIONS × 4 policies.
    baseline_res = {(s, g) for s, g, _bs in IN1K_RESOLUTIONS}
    frozen_extra = {(j.scene_size, j.canvas_grid) for j in ext_frozen
                    if (j.scene_size, j.canvas_grid) not in baseline_res}
    assert frozen_extra == {(s, g) for s, g, _bs in EXTRA_IN1K_RESOLUTIONS}


def test_in1k_scene_and_grid_appear_in_cli_args():
    """The CLI must receive --scene-size, --batch-size, and --episode.canvas-grid
    so the task picks up the right resolution; otherwise it would silently fall
    back to defaults (risk: B=64 OOM at c=64 s=1024)."""
    # Batch size per (scene, grid) comes from IN1K_RESOLUTIONS/EXTRA_IN1K_RESOLUTIONS tuples.
    bs_lookup = {(s, g): bs for s, g, bs in IN1K_RESOLUTIONS + EXTRA_IN1K_RESOLUTIONS}
    jobs = build_eval_matrix(Path("/tmp/test"), n_runs=1, n_timesteps=21,
                             tasks=["in1k-clf"], include_extra_grids=True)
    for j in jobs:
        if j.task != "in1k-clf":
            continue
        for flag, expected in [
            ("--scene-size", str(j.scene_size)),
            ("--episode.canvas-grid", str(j.canvas_grid)),
            ("--batch-size", str(bs_lookup[(j.scene_size, j.canvas_grid)])),
        ]:
            assert flag in j.args, f"{flag} missing from {j.args}"
            assert j.args[j.args.index(flag) + 1] == expected
