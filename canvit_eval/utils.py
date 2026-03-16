"""Shared utilities for evaluation.

collect_metadata is imported from canvit_probes.training.utils (single source of truth).
"""

from canvit_probes.training.utils import collect_metadata  # noqa: F401


def get_git_commit() -> str | None:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None
