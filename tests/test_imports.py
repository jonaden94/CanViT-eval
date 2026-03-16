"""Verify all public modules are importable (catches missing deps, circular imports, typos)."""


def test_config():
    from canvit_eval.config import EpisodeConfig, ade20k_root, DEFAULT_MODEL_REPO, TEACHER_REPO
    assert isinstance(DEFAULT_MODEL_REPO, str)
    assert isinstance(TEACHER_REPO, str)


def test_episode():
    from canvit_eval.episode import run_episode, EpisodeStep, Policy


def test_evaluate():
    from canvit_eval.evaluate import evaluate, MetricAccumulator, FeatureExtractor


def test_features():
    from canvit_eval.features import canvit_extractor, dinov3_extractor


def test_policies():
    from canvit_eval.policies import make_policy, PolicyName, IN1K_POLICIES, StaticPolicy, EntropyGuidedC2F
    assert len(IN1K_POLICIES) == 4


def test_runner():
    from canvit_eval.runner import eval_batches, load_model, BatchResult


def test_utils():
    from canvit_eval.utils import collect_metadata


def test_batch():
    from canvit_eval.batch import ALL_POLICIES, DETERMINISTIC, main
    assert "coarse_to_fine" in ALL_POLICIES
    assert "constant_full_scene" in DETERMINISTIC


def test_tasks():
    from canvit_eval.tasks.ade20k_seg import Config as ADE20kConfig, run as ade20k_run
    from canvit_eval.tasks.in1k_clf import Config as IN1KConfig, evaluate as in1k_evaluate
    from canvit_eval.tasks.reconstruction import Config as ReconConfig, evaluate as recon_evaluate
