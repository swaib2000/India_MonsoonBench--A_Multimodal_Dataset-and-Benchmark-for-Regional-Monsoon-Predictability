#!/usr/bin/env python3
"""
Create India-wide predictor visualizations by mosaicking state rasters.

The script reads a modeling manifest, selects a year/month band for each
modality, mosaics all available state rasters, and writes a multi-panel PNG.

Example:

MPLCONFIGDIR=/tmp/matplotlib-cache python scripts/visualize_india_modalities.py \
  --manifest modeling_manifest.csv \
  --year 2024 \
  --month 9 \
  --target-period annual \
  --include-gt \
  --output india_modality_maps/india_sep2024_annual.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for visualization") from exc

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
except Exception as exc:  # pragma: no cover
    raise SystemExit("rasterio is required for reading and mosaicking GeoTIFFs") from exc


PREDICTORS = [
    "elevation",
    "lst",
    "ndvi",
    "rh",
    "soil_moisture",
    "wind_speed",
]

PREDICTOR_TITLES = {
    "elevation": "Elevation",
    "lst": "LST",
    "ndvi": "NDVI",
    "rh": "Relative Humidity",
    "soil_moisture": "Soil Moisture",
    "wind_speed": "Wind Speed",
}

PREDICTOR_UNITS = {
    "elevation": "m",
    "lst": "K or deg C",
    "ndvi": "unitless",
    "rh": "%",
    "soil_moisture": "m3/m3",
    "wind_speed": "m/s",
}

CMAPS = {
    "elevation": "terrain",
    "lst": "inferno",
    "ndvi": "YlGn",
    "rh": "YlGnBu",
    "soil_moisture": "Blues",
    "wind_speed": "PuBuGn",
}

MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

GT_CLASS_NAMES = ["Scarcity", "Deficit", "Normal", "Excess", "Large Excess"]
GT_CMAP = ListedColormap(["#8c510a", "#d8b365", "#5ab4ac", "#2b83ba", "#762a83"])
GT_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], GT_CMAP.N)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize India-wide modality mosaics")
    parser.add_argument("--manifest", default="modeling_manifest.csv")
    parser.add_argument("--output", default="india_modality_maps/india_modalities.png")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=9)
    parser.add_argument("--target-period", default="annual", choices=["month", "annual"])
    parser.add_argument("--include-gt", action="store_true")
    parser.add_argument("--percentile-clip", default="2,98")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--interpolation",
        default="nearest",
        choices=["nearest", "bilinear"],
        help="Matplotlib display interpolation. Use bilinear for smoother figures only.",
    )
    parser.add_argument(
        "--states",
        default="",
        help="Optional comma-separated subset of states; default uses all manifest rows.",
    )
    return parser.parse_args()


def parse_percentiles(text: str) -> tuple[float, float]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if len(values) != 2:
        raise ValueError("--percentile-clip must contain two values, e.g. 2,98")
    return values[0], values[1]


def load_manifest(path: Path, states_text: str) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    if states_text.strip():
        requested = {state.strip() for state in states_text.split(",") if state.strip()}
        rows = [row for row in rows if row["state"] in requested]
        found = {row["state"] for row in rows}
        missing = requested - found
        if missing:
            raise ValueError(f"States not found in manifest: {', '.join(sorted(missing))}")
    return rows


def year_index(year: int) -> int:
    years = [2020, 2021, 2022, 2023, 2024]
    if year not in years:
        raise ValueError(f"Year must be one of {years}, got {year}")
    return years.index(year)


def predictor_band_index(path: str, predictor: str, year: int, month: int) -> int:
    with rasterio.open(path) as src:
        count = src.count
    if predictor == "elevation" or count == 1:
        return 0

    yidx = year_index(year)
    if count == 60:
        return yidx * 12 + (month - 1)
    if count == 20:
        if month not in {6, 7, 8, 9}:
            raise ValueError(
                f"{path} has 20 monsoon bands only; month {month} unavailable. "
                "Use month 6,7,8,9 or a full-year manifest."
            )
        return yidx * 4 + [6, 7, 8, 9].index(month)
    raise ValueError(f"Unsupported predictor band count for {path}: {count}")


def gt_band_index(path: str, year: int, month: int, target_period: str) -> int:
    with rasterio.open(path) as src:
        count = src.count
    yidx = year_index(year)
    if count == 65:
        return yidx * 13 + 12 if target_period == "annual" else yidx * 13 + (month - 1)
    if count == 25:
        periods = [6, 7, 8, 9, "annual"]
        period = "annual" if target_period == "annual" else month
        if period not in periods:
            raise ValueError(f"{path} has 25 monsoon/annual bands only; target {period} unavailable.")
        return yidx * 5 + periods.index(period)
    raise ValueError(f"Unsupported GT band count for {path}: {count}")


def mosaic_band(paths_and_bands: list[tuple[str, int]]) -> np.ndarray:
    sources = []
    memory_files = []
    try:
        for path, band_idx in paths_and_bands:
            with rasterio.open(path) as src:
                arr = src.read(band_idx + 1)
                profile = src.profile.copy()
                profile.update(count=1)
                memfile = MemoryFile()
                with memfile.open(**profile) as dataset:
                    dataset.write(arr, 1)
                memory_files.append(memfile)
                sources.append(memfile.open())
        mosaic, _transform = merge(sources, indexes=[1], method="first")
        arr = mosaic[0].astype(np.float32)
        nodata_values = [src.nodata for src in sources if src.nodata is not None]
        for nodata in nodata_values:
            arr = np.where(arr == nodata, np.nan, arr)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return arr
    finally:
        for src in sources:
            src.close()
        for memfile in memory_files:
            memfile.close()


def finite_limits(arr: np.ndarray, p_low: float, p_high: float) -> tuple[float | None, float | None]:
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return None, None
    low, high = np.nanpercentile(values, [p_low, p_high])
    if np.isclose(low, high):
        return float(np.nanmin(values)), float(np.nanmax(values))
    return float(low), float(high)


def make_panels(rows: list[dict[str, str]], args: argparse.Namespace) -> list[tuple[str, str, np.ndarray, str, bool]]:
    panels = []
    for predictor in PREDICTORS:
        paths_and_bands = []
        for row in rows:
            band_idx = predictor_band_index(row[predictor], predictor, args.year, args.month)
            paths_and_bands.append((row[predictor], band_idx))
        arr = mosaic_band(paths_and_bands)
        panels.append((predictor, PREDICTOR_TITLES[predictor], arr, CMAPS[predictor], False))

    if args.include_gt:
        paths_and_bands = []
        for row in rows:
            band_idx = gt_band_index(row["gt"], args.year, args.month, args.target_period)
            paths_and_bands.append((row["gt"], band_idx))
        arr = mosaic_band(paths_and_bands)
        month_name = MONTH_NAMES[args.month]
        title = "GT Annual Class" if args.target_period == "annual" else f"GT {month_name} Class"
        panels.append(("gt", title, arr, "gt", True))
    return panels


def plot_panels(panels: list[tuple[str, str, np.ndarray, str, bool]], args: argparse.Namespace) -> None:
    p_low, p_high = parse_percentiles(args.percentile_clip)
    cols = 4 if len(panels) > 4 else len(panels)
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 5.0 * rows), squeeze=False)
    month_name = MONTH_NAMES[args.month]
    fig.suptitle(
        f"India-Wide Multimodal Raster Mosaics: {month_name} {args.year}",
        fontsize=16,
        fontweight="bold",
    )

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (key, title, arr, cmap, is_gt) in zip(axes.ravel(), panels):
        ax.axis("off")
        if is_gt:
            image = ax.imshow(arr, cmap=GT_CMAP, norm=GT_NORM, interpolation="nearest")
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 1, 2, 3, 4])
            cbar.ax.set_yticklabels(GT_CLASS_NAMES, fontsize=7)
            cbar.set_label("Rainfall anomaly class", fontsize=8)
        else:
            vmin, vmax = finite_limits(arr, p_low, p_high)
            image = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=args.interpolation)
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label(PREDICTOR_UNITS[key], fontsize=8)
        ax.set_title(title, fontsize=11)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    if args.month < 1 or args.month > 12:
        raise ValueError("--month must be between 1 and 12")
    rows = load_manifest(Path(args.manifest), args.states)
    if not rows:
        raise SystemExit("No states selected from manifest")
    panels = make_panels(rows, args)
    plot_panels(panels, args)


if __name__ == "__main__":
    main()
