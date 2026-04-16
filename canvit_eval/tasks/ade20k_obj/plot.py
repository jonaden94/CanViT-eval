"""Plot DINOv3 per-class IoU vs object size (% area) by resolution.

Reads output/dv3_ade20k_per_image.parquet produced by dv3_ade20k_per_image.py
and produces:
  Panel A — LOWESS IoU vs object area with 95% bootstrap CI, one band per resolution
  Panel B — per-resolution scatter with LOWESS overlay
  Panel C — LOWESS of 512px − 128px IoU difference with 95% bootstrap CI

Output: output/dv3_obj_size.{svg,png}

Usage:
    uv run python canvit_eval/tasks/ade20k_obj/plot.py
    uv run python canvit_eval/tasks/ade20k_obj/plot.py --n-boot 500 --output output/fig
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tyro
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess


@dataclass
class Config:
    input: Path = Path("output/dv3_ade20k_per_image.parquet")
    output: Path = Path("output/dv3_obj_size")
    frac: float = 0.25
    """LOWESS bandwidth fraction."""
    n_boot: int = 200
    """Bootstrap iterations for 95% CI bands."""
    n_grid: int = 300
    """Number of x points to evaluate LOWESS on."""
    scatter_sample: int = 5_000
    """Max points per resolution in the scatter panel."""


def _lowess_ci(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    frac: float,
    n_boot: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit LOWESS and bootstrap 95% CI on x_grid.

    Returns (smooth, ci_lo, ci_hi), all shape [len(x_grid)].
    """
    delta = 0.005 * (x_grid[-1] - x_grid[0])

    def _fit(xs, ys):
        out = sm_lowess(ys, xs, frac=frac, delta=delta, return_sorted=True)
        return np.interp(x_grid, out[:, 0], out[:, 1])

    smooth = _fit(x, y)

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(x_grid)))
    for b in range(n_boot):
        idx = rng.integers(len(x), size=len(x))
        boot[b] = _fit(x[idx], y[idx])

    return smooth, np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)


def _resolution_cmap(resolutions: list[int]) -> dict[int, tuple]:
    cmap = plt.get_cmap("plasma")
    n = len(resolutions)
    return {r: cmap(i / (n - 1)) for i, r in enumerate(resolutions)}


def plot(cfg: Config) -> None:
    df = pd.read_parquet(cfg.input)
    resolutions = sorted(df["resolution"].unique().tolist())
    colors = _resolution_cmap(resolutions)

    x_min = df["area"].min()
    x_max = df["area"].max()
    x_grid = np.linspace(x_min, x_max, cfg.n_grid)

    print(f"Fitting LOWESS + {cfg.n_boot} bootstrap iterations per curve...")

    # Precompute LOWESS + CI for each resolution
    curves: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for res in resolutions:
        sub = df[df["resolution"] == res]
        x, y = sub["area"].to_numpy(), sub["iou"].to_numpy()
        print(f"  {res}px  (n={len(x):,})", flush=True)
        curves[res] = _lowess_ci(x, y, x_grid, cfg.frac, cfg.n_boot, seed=res)

    # Panel C: LOWESS on per-observation 512 − 128 diff
    # Inner join on (image_idx, class_idx) so we diff the same observations
    df128 = df[df["resolution"] == 128][["image_idx", "class_idx", "area", "iou"]].rename(columns={"iou": "iou_128"})
    df512 = df[df["resolution"] == 512][["image_idx", "class_idx", "iou"]].rename(columns={"iou": "iou_512"})
    diff_df = df128.merge(df512, on=["image_idx", "class_idx"])
    diff_df["diff"] = diff_df["iou_512"] - diff_df["iou_128"]
    x_diff = diff_df["area"].to_numpy()
    y_diff = diff_df["diff"].to_numpy()
    print(f"  diff 512-128  (n={len(x_diff):,})", flush=True)
    diff_smooth, diff_lo, diff_hi = _lowess_ci(x_diff, y_diff, x_grid, cfg.frac, cfg.n_boot, seed=0)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("DINOv3 segmentation IoU vs object size", fontsize=13)
    pct_fmt = mticker.PercentFormatter(xmax=1, decimals=1)

    # ── Panel A: LOWESS + CI bands by resolution ────────────────────────────
    ax = axes[0]
    for res in resolutions:
        smooth, ci_lo, ci_hi = curves[res]
        ax.plot(x_grid, smooth, color=colors[res], linewidth=1.5, label=f"{res}px")
        ax.fill_between(x_grid, ci_lo, ci_hi, color=colors[res], alpha=0.15)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.set_xlabel("Object area (% of image)")
    ax.set_ylabel("Mean IoU")
    ax.set_title("LOWESS IoU vs object size")
    ax.legend(title="Resolution", fontsize=8, title_fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel B: scatter + LOWESS curve per resolution ──────────────────────
    ax = axes[1]
    rng = np.random.default_rng(0)
    for res in resolutions:
        sub = df[df["resolution"] == res]
        sample = sub.sample(min(cfg.scatter_sample, len(sub)), random_state=rng.integers(1 << 31))
        ax.scatter(sample["area"], sample["iou"], alpha=0.05, s=4,
                   color=colors[res], rasterized=True)
        smooth, _, _ = curves[res]
        ax.plot(x_grid, smooth, color=colors[res], linewidth=1.5, label=f"{res}px")
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.set_xlabel("Object area (% of image)")
    ax.set_ylabel("IoU")
    ax.set_title("Scatter + LOWESS")
    ax.legend(title="Resolution", fontsize=8, title_fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel C: 512px − 128px LOWESS diff + CI ─────────────────────────────
    ax = axes[2]
    ax.plot(x_grid, diff_smooth, color="steelblue", linewidth=1.8)
    ax.fill_between(x_grid, diff_lo, diff_hi, color="steelblue", alpha=0.2, label="95% CI")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.set_xlabel("Object area (% of image)")
    ax.set_ylabel("IoU difference (512px − 128px)")
    ax.set_title("Resolution gain: 512px vs 128px")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.output.with_suffix(".svg"), dpi=150)
    fig.savefig(cfg.output.with_suffix(".png"), dpi=150)
    print(f"Saved {cfg.output}.svg  +  .png")


if __name__ == "__main__":
    plot(tyro.cli(Config))
