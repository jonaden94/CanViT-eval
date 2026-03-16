"""Shared utilities for evaluation."""

import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import torch


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def collect_metadata(cfg: Any) -> dict:
    """Collect run metadata for reproducibility."""
    return {
        "config": asdict(cfg),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("HOSTNAME") or os.environ.get("SLURMD_NODENAME"),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
