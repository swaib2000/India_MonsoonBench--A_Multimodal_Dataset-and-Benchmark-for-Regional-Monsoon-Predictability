#!/usr/bin/env python3
"""
Create professional README figures for India MonsoonBench.

The script reuses the India-wide raster mosaic utilities to generate:
1. One clean PNG per predictor/modality.
2. A wide rectangular collage suitable for the GitHub README.

Example:

MPLCONFIGDIR=/tmp/matplotlib-cache python create_readme_modality_collage.py \
  --manifest modeling_manifest.csv \
  --year 2024 \
  --month 9 \
  --target-period annual \
  --output-dir website/assets/images/readme
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from visualize_india_modalities import (
    CMAPS,
    GT_CLASS_NAMES,
    GT_CMAP,
    GT_NORM,
    MONTH_NAMES,
    PREDICTOR_TITLES,
    finite_limits,
    load_manifest,
    make_panels,
    parse_percentiles,
)


DISPLAY_NAMES = {
    "elevation": "Elevation",
    "lst": "Land Surface Temperature",
    "ndvi": "NDVI",
    "rh": "Relative Humidity",
    "soil_moisture": "Soil Moisture",
    "wind_speed": "Wind Speed",
    "gt": "Rainfall Anomaly Class",
}

DISPLAY_UNITS = {
    "elevation": "m",
    "lst": "deg C",
    "ndvi": "index",
    "rh": "%",
    "soil_moisture": "m3/m3",
    "wind_speed": "m/s",
    "gt": "IMD LPA class",
}

PANEL_ORDER = [
    "elevation",
    "lst",
    "ndvi",
    "rh",
    "soil_moisture",
    "wind_speed",
    "gt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README modality panels and collage")
    parser.add_argument("--manifest", default="modeling_manifest.csv")
    parser.add_argument("--output-dir", default="website/assets/images/readme")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=9)
    parser.add_argument("--target-period", default="annual", choices=["month", "annual"])
    parser.add_argument("--percentile-clip", default="2,98")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--states", default="", help="Optional comma-separated state subset")
    return parser.parse_args()


def ordered_panels(panels: list[tuple[str, str, np.ndarray, str, bool]]) -> list[tuple[str, str, np.ndarray, str, bool]]:
    by_key = {panel[0]: panel for panel in panels}
    return [by_key[key] for key in PANEL_ORDER if key in by_key]


def draw_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    key: str,
    arr: np.ndarray,
    is_gt: bool,
    p_low: float,
    p_high: float,
    interpolation: str = "bilinear",
) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if is_gt:
        image = ax.imshow(arr, cmap=GT_CMAP, norm=GT_NORM, interpolation="nearest")
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(GT_CLASS_NAMES, fontsize=6)
        cbar.ax.tick_params(length=0)
        cbar.outline.set_linewidth(0.4)
    else:
        vmin, vmax = finite_limits(arr, p_low, p_high)
        image = ax.imshow(
            arr,
            cmap=CMAPS[key],
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.015)
        cbar.ax.tick_params(labelsize=6, length=2, width=0.4)
        cbar.outline.set_linewidth(0.4)

    ax.set_title(DISPLAY_NAMES.get(key, PREDICTOR_TITLES.get(key, key)), fontsize=10, fontweight="bold", pad=7)
    ax.text(
        0.02,
        0.04,
        DISPLAY_UNITS.get(key, ""),
        transform=ax.transAxes,
        fontsize=7,
        color="#263238",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cfd8dc", "alpha": 0.88},
    )


def save_individual_panels(
    panels: list[tuple[str, str, np.ndarray, str, bool]],
    output_dir: Path,
    p_low: float,
    p_high: float,
    dpi: int,
) -> None:
    panel_dir = output_dir / "individual_modalities"
    panel_dir.mkdir(parents=True, exist_ok=True)

    for key, _title, arr, _cmap, is_gt in panels:
        fig, ax = plt.subplots(figsize=(5.0, 4.7), facecolor="white")
        draw_panel(fig, ax, key, arr, is_gt, p_low, p_high)
        fig.tight_layout(pad=0.6)
        path = panel_dir / f"{key}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Wrote {path}")


def save_collage(
    panels: list[tuple[str, str, np.ndarray, str, bool]],
    output_dir: Path,
    p_low: float,
    p_high: float,
    year: int,
    month: int,
    dpi: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18.0, 8.0), facecolor="#f8faf7")
    grid = fig.add_gridspec(
        2,
        4,
        left=0.03,
        right=0.985,
        bottom=0.06,
        top=0.84,
        wspace=0.11,
        hspace=0.20,
    )

    fig.text(
        0.5,
        0.955,
        "India MonsoonBench",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#17202a",
    )
    month_name = MONTH_NAMES[month]
    fig.text(
        0.5,
        0.905,
        f"Spatially aligned multimodal Earth-observation predictors for rainfall anomaly benchmarking ({month_name} {year})",
        ha="center",
        va="center",
        fontsize=12,
        color="#455a64",
    )

    axes = [fig.add_subplot(grid[idx // 4, idx % 4]) for idx in range(8)]
    for ax in axes:
        ax.set_facecolor("white")

    for ax, (key, _title, arr, _cmap, is_gt) in zip(axes[:7], panels):
        draw_panel(fig, ax, key, arr, is_gt, p_low, p_high)

    info_ax = axes[7]
    info_ax.axis("off")
    info_ax.text(
        0.02,
        0.86,
        "Benchmark task",
        fontsize=14,
        fontweight="bold",
        color="#17202a",
        transform=info_ax.transAxes,
    )
    info_text = (
        "Monthly autoregressive forecasting\n\n"
        "Input: 3 consecutive months\n"
        "Target: next-month rainfall class\n\n"
        "X(s,t-2), X(s,t-1), X(s,t)\n"
        "-> Y(s,t+1)\n\n"
        "19 channels = 6 dynamic predictors\n"
        "x 3 months + elevation"
    )
    info_ax.text(
        0.02,
        0.74,
        info_text,
        fontsize=9.4,
        color="#37474f",
        linespacing=1.35,
        transform=info_ax.transAxes,
        va="top",
    )
    info_ax.text(
        0.02,
        0.04,
        "Ground truth: CHIRPS rainfall categories\ncomputed against IMD LPA normals.",
        fontsize=8.1,
        color="#607d8b",
        linespacing=1.25,
        transform=info_ax.transAxes,
        va="bottom",
    )

    output = output_dir / "monsoonbench_readme_collage.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="#f8faf7")
    plt.close(fig)
    print(f"Wrote {output}")
    return output


def main() -> None:
    args = parse_args()
    rows = load_manifest(Path(args.manifest), args.states)
    if not rows:
        raise SystemExit("No rows selected from manifest")

    panel_args = argparse.Namespace(
        year=args.year,
        month=args.month,
        target_period=args.target_period,
        include_gt=True,
    )
    panels = ordered_panels(make_panels(rows, panel_args))
    p_low, p_high = parse_percentiles(args.percentile_clip)
    output_dir = Path(args.output_dir)

    save_individual_panels(panels, output_dir, p_low, p_high, args.dpi)
    save_collage(panels, output_dir, p_low, p_high, args.year, args.month, args.dpi)


if __name__ == "__main__":
    main()
