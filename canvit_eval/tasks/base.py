"""Shared base for task Configs.

Every eval task has the same runtime infrastructure (output path, device,
DataLoader batch/worker sizes, AMP). Task-specific fields come on top.

`kw_only=True` lets subclasses add required fields after the base's defaults
without dataclass inheritance ordering complaints.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class TaskConfig:
    output: Path
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True
