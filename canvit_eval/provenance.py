"""Reproducibility metadata for eval sidecars."""

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import torch


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return ""


def git_commit() -> str:
    return _git("rev-parse", "HEAD") or "unknown"


def git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"


def git_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def provenance() -> dict[str, Any]:
    """Env-level reproducibility dict. Task-specific fields (input paths, hyper-
    params, GPU name) are added alongside by each task's sidecar writer."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty": git_dirty(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cmdline": " ".join(sys.argv),
    }


def device_info(device: torch.device | None) -> dict[str, Any]:
    """`device` + (on CUDA) `cuda_device_name`. Callers splat into their sidecar."""
    if device is None:
        return {}
    info: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
    return info
