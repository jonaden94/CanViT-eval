"""Unit tests for the pretraining view-scale resolution (foveated-scale footgun fix)."""

from canvit_eval.config import resolve_scale_from_metadata


def _meta(mode: str, fixed_scale=None, patcher="foveated") -> dict:
    return {"pretrain_view_scale": {"patcher_name": patcher, "mode": mode, "fixed_scale": fixed_scale}}


def test_explicit_override_always_wins():
    # Even with metadata present, a user-set override_scale is respected.
    scale, reason = resolve_scale_from_metadata(_meta("fixed", 2.0), override_scale=0.5)
    assert scale == 0.5
    assert "respecting user" in reason


def test_no_metadata_is_noop():
    # All pre-fix repos: unchanged behavior (policy scales).
    assert resolve_scale_from_metadata({}, None) == (None, resolve_scale_from_metadata({}, None)[1])
    scale, _ = resolve_scale_from_metadata({}, None)
    assert scale is None
    assert resolve_scale_from_metadata(None, None)[0] is None


def test_fixed_foveated_pins_scale():
    scale, reason = resolve_scale_from_metadata(_meta("fixed", 2.0), None)
    assert scale == 2.0
    assert "FIXED" in reason and "2.0" in reason


def test_fixed_square_pins_scale():
    scale, _ = resolve_scale_from_metadata(_meta("fixed", 1.41, patcher="square"), None)
    assert scale == 1.41


def test_multiscale_lets_policy_through():
    for mode in ("per_rollout", "per_glimpse"):
        scale, reason = resolve_scale_from_metadata(_meta(mode, None), None)
        assert scale is None
        assert "multi-scale" in reason


def test_fixed_without_value_falls_back_to_none():
    scale, reason = resolve_scale_from_metadata(_meta("fixed", None), None)
    assert scale is None
    assert "no fixed_scale" in reason


def test_uniform_pretrain_view_scale_none_is_noop():
    # Converter writes pretrain_view_scale=None for uniform models.
    scale, _ = resolve_scale_from_metadata({"pretrain_view_scale": None}, None)
    assert scale is None
