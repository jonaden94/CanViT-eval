"""Run metadata for eval-result sidecars.

Ported verbatim from the (now-archived) CanViT-specialize `training/utils.py` so
eval owns its own sidecar format and needs no dependency on the training repos
for a git+timestamp helper. This embeds the full config (``asdict(cfg)``), unlike
``provenance.provenance()`` which deliberately omits identity-revealing fields for
anonymized bundles — use ``provenance()`` for shippable sidecars, this for local
run records.
"""

import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import torch


def collect_metadata(cfg: Any) -> dict[str, Any]:
    """Collect portable metadata for saved evaluation/training artifacts."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None

    return {
        "config": asdict(cfg),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
