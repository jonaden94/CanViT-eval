"""Plot DINOv3 and CanViT per-class IoU vs object size (% area).

3-row × 4-column figure:
  Row 0: DINOv3          — lines = resolution (128..512)
  Row 1: CanViT (t=0)    — lines = canvas_resolution, first glimpse
  Row 2: CanViT (t=last) — lines = canvas_resolution, final glimpse

  Col 0: LOWESS IoU + 95% bootstrap CI
  Col 1: Scatter + LOWESS
  Col 2: IoU diff within model (hi − lo resolution)
  Col 3: CanViT − DINOv3 (empty for DINOv3 row)

Col 3 lines are controlled by --canvit-vs-dino-pairs, formatted as "canvas_res:dino_res",
e.g. "64:512" "64:256" "32:512" gives three lines per CanViT row.

Usage:
    uv run python canvit_eval/tasks/ade20k_obj/plot.py
    uv run python canvit_eval/tasks/ade20k_obj/plot.py \\
        --canvit-vs-dino-pairs 64:512 64:256 32:512
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tyro
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess


@dataclass
class Config:
    input_dv3: Path = Path("output/dv3_ade20k_per_image.parquet")
    input_canvit: Path = Path("output/canvit_ade20k_per_image.parquet")
    output: Path = Path("output/dv3_vs_canvit_obj_size")
    canvit_vs_dino_pairs: list[str] = field(default_factory=lambda: ["64:512"])
    """CanViT−DINOv3 comparison pairs for col 3, as 'canvas_res:dino_res'."""
    frac: float = 0.25
    n_boot: int = 200
    n_grid: int = 300
    scatter_sample: int = 5_000


def _parse_pairs(pairs: list[str]) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in p.split(":")) for p in pairs]


# ── LOWESS + bootstrap CI ─────────────────────────────────────────────────────


def _lowess_ci(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    frac: float,
    n_boot: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LOWESS smooth + 95% bootstrap CI evaluated on x_grid."""
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


# ── Curve computation ─────────────────────────────────────────────────────────


def _compute_row_curves(
    df: pd.DataFrame,
    group_col: str,
    lo_val: int,
    hi_val: int,
    x_grid: np.ndarray,
    frac: float,
    n_boot: int,
) -> tuple[dict[int, tuple], tuple]:
    groups = sorted(df[group_col].unique().tolist())
    curves: dict[int, tuple] = {}
    for g in groups:
        sub = df[df[group_col] == g]
        print(f"    {group_col}={g}  n={len(sub):,}", flush=True)
        curves[g] = _lowess_ci(sub["area"].to_numpy(), sub["iou"].to_numpy(), x_grid, frac, n_boot, seed=g)

    df_lo = df[df[group_col] == lo_val][["image_idx", "class_idx", "area", "iou"]].rename(columns={"iou": "iou_lo"})
    df_hi = df[df[group_col] == hi_val][["image_idx", "class_idx", "iou"]].rename(columns={"iou": "iou_hi"})
    diff_df = df_lo.merge(df_hi, on=["image_idx", "class_idx"])
    diff_df["diff"] = diff_df["iou_hi"] - diff_df["iou_lo"]
    print(f"    diff {hi_val}−{lo_val}  n={len(diff_df):,}", flush=True)
    diff_curves = _lowess_ci(diff_df["area"].to_numpy(), diff_df["diff"].to_numpy(), x_grid, frac, n_boot, seed=0)

    return curves, diff_curves


def _compute_canvit_vs_dino_curves(
    canvit_t: pd.DataFrame,
    dv3_df: pd.DataFrame,
    pairs: list[tuple[int, int]],
    x_grid: np.ndarray,
    frac: float,
    n_boot: int,
) -> dict[tuple[int, int], tuple]:
    """LOWESS+CI for each (canvas_res, dino_res) pair on per-observation IoU diff."""
    curves: dict[tuple[int, int], tuple] = {}
    for canvas_res, dino_res in pairs:
        cv = canvit_t[canvit_t["canvas_resolution"] == canvas_res][
            ["image_idx", "class_idx", "area", "iou"]
        ].rename(columns={"iou": "iou_cv"})
        dv = dv3_df[dv3_df["resolution"] == dino_res][
            ["image_idx", "class_idx", "iou"]
        ].rename(columns={"iou": "iou_dv"})
        merged = cv.merge(dv, on=["image_idx", "class_idx"])
        merged["diff"] = merged["iou_cv"] - merged["iou_dv"]
        print(f"    c{canvas_res} − {dino_res}px  n={len(merged):,}", flush=True)
        curves[(canvas_res, dino_res)] = _lowess_ci(
            merged["area"].to_numpy(), merged["diff"].to_numpy(),
            x_grid, frac, n_boot, seed=(canvas_res * 1000 + dino_res) % (2**32),
        )
    return curves


# ── Drawing ───────────────────────────────────────────────────────────────────


def _line_cmap(values: list[int]) -> dict[int, tuple]:
    cmap = plt.get_cmap("plasma")
    n = len(values)
    return {v: cmap(i / max(n - 1, 1)) for i, v in enumerate(values)}


def _draw_row(
    axes: tuple,
    df: pd.DataFrame,
    group_col: str,
    lo_val: int,
    hi_val: int,
    curves: dict[int, tuple],
    diff_curves: tuple,
    x_grid: np.ndarray,
    x_min: float,
    x_max: float,
    cfg: Config,
    rng: "np.random.Generator",
    row_title: str,
    diff_ylabel: str,
    diff_title: str,
) -> None:
    pct_fmt = mticker.PercentFormatter(xmax=1, decimals=1)
    groups = sorted(curves.keys())
    colors = _line_cmap(groups)

    def _label(v: int) -> str:
        return f"c{v}" if group_col == "canvas_resolution" else f"{v}px"

    # Col 0: LOWESS + CI bands
    ax = axes[0]
    for g in groups:
        smooth, ci_lo, ci_hi = curves[g]
        ax.plot(x_grid, smooth, color=colors[g], linewidth=1.5, label=_label(g))
        ax.fill_between(x_grid, ci_lo, ci_hi, color=colors[g], alpha=0.15)
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.set_xlabel("Object area (% of image)")
    ax.set_ylabel("IoU")
    ax.set_title(f"{row_title} — LOWESS IoU vs object size")
    ax.legend(title="Resolution", fontsize=8, title_fontsize=8)
    ax.grid(True, alpha=0.3)

    # Col 1: scatter + LOWESS
    ax = axes[1]
    for g in groups:
        sub = df[df[group_col] == g]
        sample = sub.sample(min(cfg.scatter_sample, len(sub)), random_state=rng.integers(1 << 31))
        ax.scatter(sample["area"], sample["iou"], alpha=0.05, s=4, color=colors[g], rasterized=True)
        smooth, _, _ = curves[g]
        ax.plot(x_grid, smooth, color=colors[g], linewidth=1.5, label=_label(g))
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.set_xlabel("Object area (% of image)")
    ax.set_ylabel("IoU")
    ax.set_title(f"{row_title} — Scatter + LOWESS")
    ax.legend(title="Resolution", fontsize=8, title_fontsize=8)
    ax.grid(True, alpha=0.3)

    # Col 2: within-model diff
    ax = axes[2]
    smooth, ci_lo, ci_hi = diff_curves
    ax.plot(x_grid, smooth, color="steelblue", linewidth=1.8)
    ax.fill_between(x_grid, ci_lo, ci_hi, color="steelblue", alpha=0.2, label="95% CI")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.set_xlabel("Object area (% of image)")
    ax.set_ylabel(diff_ylabel)
    ax.set_title(diff_title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _draw_canvit_vs_dino_panel(
    ax,
    cv_dino_curves: dict[tuple[int, int], tuple],
    x_grid: np.ndarray,
    x_min: float,
    x_max: float,
    row_title: str,
) -> None:
    pct_fmt = mticker.PercentFormatter(xmax=1, decimals=1)
    n = len(cv_dino_curves)
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    for i, ((canvas_res, dino_res), (smooth, ci_lo, ci_hi)) in enumerate(sorted(cv_dino_curves.items())):
        color = cmap(i / max(n - 1, 1))
        label = f"c{canvas_res} − {dino_res}px"
        ax.plot(x_grid, smooth, color=color, linewidth=1.5, label=label)
        ax.fill_between(x_grid, ci_lo, ci_hi, color=color, alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.set_xlabel("Object area (% of image)")
    ax.set_ylabel("IoU diff (CanViT − DINOv3)")
    ax.set_title(f"{row_title} − DINOv3")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ── Main ──────────────────────────────────────────────────────────────────────


def plot(cfg: Config) -> None:
    pairs = _parse_pairs(cfg.canvit_vs_dino_pairs)

    dv3_df = pd.read_parquet(cfg.input_dv3)
    dv3_df["iou"] = dv3_df["inter_px"] / dv3_df["union_px"]
    canvit_df = pd.read_parquet(cfg.input_canvit)
    canvit_df["iou"] = canvit_df["inter_px"] / canvit_df["union_px"]

    t_last = canvit_df["timestep"].max()
    canvit_t0   = canvit_df[canvit_df["timestep"] == 0].copy()
    canvit_tlast = canvit_df[canvit_df["timestep"] == t_last].copy()

    all_areas = pd.concat([dv3_df["area"], canvit_t0["area"], canvit_tlast["area"]])
    x_min, x_max = all_areas.min(), all_areas.max()
    x_grid = np.linspace(x_min, x_max, cfg.n_grid)

    dv3_resolutions = sorted(dv3_df["resolution"].unique().tolist())
    dv3_lo, dv3_hi = dv3_resolutions[0], dv3_resolutions[-1]
    canvit_resolutions = sorted(canvit_df["canvas_resolution"].unique().tolist())
    cv_lo, cv_hi = canvit_resolutions[0], canvit_resolutions[-1]

    print(f"Fitting LOWESS ({cfg.n_boot} bootstrap iterations per curve)...")

    print("  [DINOv3]")
    dv3_curves, dv3_diff = _compute_row_curves(
        dv3_df, "resolution", dv3_lo, dv3_hi, x_grid, cfg.frac, cfg.n_boot,
    )
    print("  [CanViT t=0]")
    cv0_curves, cv0_diff = _compute_row_curves(
        canvit_t0, "canvas_resolution", cv_lo, cv_hi, x_grid, cfg.frac, cfg.n_boot,
    )
    print(f"  [CanViT t={t_last}]")
    cvlast_curves, cvlast_diff = _compute_row_curves(
        canvit_tlast, "canvas_resolution", cv_lo, cv_hi, x_grid, cfg.frac, cfg.n_boot,
    )

    print("  [CanViT − DINOv3, t=0]")
    cv0_vs_dino = _compute_canvit_vs_dino_curves(
        canvit_t0, dv3_df, pairs, x_grid, cfg.frac, cfg.n_boot,
    )
    print(f"  [CanViT − DINOv3, t={t_last}]")
    cvlast_vs_dino = _compute_canvit_vs_dino_curves(
        canvit_tlast, dv3_df, pairs, x_grid, cfg.frac, cfg.n_boot,
    )

    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    fig.suptitle("Segmentation IoU vs object size: DINOv3 vs CanViT", fontsize=14)
    rng = np.random.default_rng(42)

    _draw_row(
        axes[0, :3], dv3_df, "resolution", dv3_lo, dv3_hi,
        dv3_curves, dv3_diff, x_grid, x_min, x_max, cfg, rng,
        row_title="DINOv3",
        diff_ylabel=f"IoU diff ({dv3_hi}px − {dv3_lo}px)",
        diff_title=f"Resolution gain: {dv3_hi}px vs {dv3_lo}px",
    )
    axes[0, 3].set_visible(False)

    _draw_row(
        axes[1, :3], canvit_t0, "canvas_resolution", cv_lo, cv_hi,
        cv0_curves, cv0_diff, x_grid, x_min, x_max, cfg, rng,
        row_title="CanViT (t=0)",
        diff_ylabel=f"IoU diff (c{cv_hi} − c{cv_lo})",
        diff_title=f"Canvas gain: c{cv_hi} vs c{cv_lo}",
    )
    _draw_canvit_vs_dino_panel(
        axes[1, 3], cv0_vs_dino, x_grid, x_min, x_max, row_title="CanViT (t=0)",
    )

    _draw_row(
        axes[2, :3], canvit_tlast, "canvas_resolution", cv_lo, cv_hi,
        cvlast_curves, cvlast_diff, x_grid, x_min, x_max, cfg, rng,
        row_title=f"CanViT (t={t_last})",
        diff_ylabel=f"IoU diff (c{cv_hi} − c{cv_lo})",
        diff_title=f"Canvas gain: c{cv_hi} vs c{cv_lo}",
    )
    _draw_canvit_vs_dino_panel(
        axes[2, 3], cvlast_vs_dino, x_grid, x_min, x_max, row_title=f"CanViT (t={t_last})",
    )

    fig.tight_layout()
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.output.with_suffix(".svg"), dpi=150)
    fig.savefig(cfg.output.with_suffix(".png"), dpi=150)
    print(f"Saved {cfg.output}.svg  +  .png")


if __name__ == "__main__":
    plot(tyro.cli(Config))
