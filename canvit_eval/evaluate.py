"""THE evaluation function. One function, all tasks.

Adding a new task = defining a new MetricAccumulator.
No new boilerplate for dataset loading, model loading, episode running, or saving.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from canvit_eval.utils import collect_metadata

log = logging.getLogger(__name__)


class MetricAccumulator(Protocol):
    """Per-task metric computation. Implement this to add a new eval task.

    Called once per batch per timestep with the extracted features.
    """

    def update(self, t: int, features: Tensor, batch: tuple) -> None:
        """Accumulate metrics for timestep t.

        Args:
            t: Timestep index.
            features: Extracted features for this timestep. Shape depends on extractor.
            batch: Raw dataloader batch (images, masks) or (images, labels) etc.
        """
        ...

    def compute(self) -> dict:
        """Compute final metrics. Called once after all batches."""
        ...


# images in, per-timestep features out
FeatureExtractor = Callable[[Tensor], list[Tensor]]


@torch.inference_mode()
def evaluate(
    *,
    loader: DataLoader,
    extract_features: FeatureExtractor,
    metric: MetricAccumulator,
    output: Path,
    device: torch.device,
    amp: bool = True,
    metadata: dict | None = None,
) -> Path:
    """Run evaluation. THE one function for all tasks.

    Args:
        loader: DataLoader (first element of each batch = images).
        extract_features: Images → list of per-timestep feature tensors.
        metric: Task-specific accumulator (IoU, accuracy, cosine sim, etc.).
        output: Path to save .pt results.
        device: Compute device.
        amp: Automatic mixed precision.
        metadata: Extra metadata to save alongside results.
    """
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if amp else torch.float32
    t_start = time.monotonic()
    n_images = 0

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            n_images += images.shape[0]

            features_per_t = extract_features(images)
            for t, features in enumerate(features_per_t):
                metric.update(t, features, batch)

    wall_time = time.monotonic() - t_start
    results = {
        **metric.compute(),
        "metadata": {**(metadata or {}), "wall_time_seconds": wall_time, "n_images": n_images},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output)
    log.info("Saved to %s (%.1fs, %d images)", output, wall_time, n_images)
    return output
