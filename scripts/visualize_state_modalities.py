#!/usr/bin/env python3
"""
Visualize multimodal raster maps for each state.

Examples:

python scripts/visualize_state_modalities.py \
  --manifest modeling_manifest.csv \
  --states "Andhra Pradesh,Uttarakhand" \
  --year 2024 \
  --month 9 \
  --include-gt

python scripts/visualize_state_modalities.py \
  --manifest modeling_manifest_full_year_states.csv \
  --all-states \
  --year 2024 \
  --month 6 \
  --output-dir modality_maps_jun2024
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
except Exception as exc:  # pragma: no cover
    raise SystemExit("rasterio is required for reading GeoTIFFs") from exc


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
    parser = argparse.ArgumentParser(description="Visualize state modality raster maps")
    parser.add_argument("--manifest", default="modeling_manifest.csv")
    parser.add_argument("--output-dir", default="state_modality_maps")
    parser.add_argument("--states", default="", help="Comma-separated state names. Ignored if --all-states is set.")
    parser.add_argument("--all-states", action="store_true")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=9)
    parser.add_argument(
        "--target-period",
        default="month",
        choices=["month", "annual"],
        help="GT band to show when --include-gt is used",
    )
    parser.add_argument("--include-gt", action="store_true")
    parser.add_argument(
        "--percentile-clip",
        default="2,98",
        help="Percentile range for continuous map color scaling, e.g. 2,98",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--max-states",
        type=int,
        default=0,
        help="Optional cap for quick previews; 0 means no cap",
    )
    return parser.parse_args()


def parse_percentiles(text: str) -> tuple[float, float]:
    parts = [float(token.strip()) for token in text.split(",") if token.strip()]
    if len(parts) != 2:
        raise ValueError("--percentile-clip must contain two values, e.g. 2,98")
    return parts[0], parts[1]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def year_index(year: int, years: list[int]) -> int:
    if year not in years:
        raise ValueError(f"Year {year} not found in supported years: {years}")
    return years.index(year)


def predictor_band_index(path: str, predictor: str, year: int, month: int) -> int:
    with rasterio.open(path) as src:
        count = src.count

    if predictor == "elevation":
        return 0

    years = [2020, 2021, 2022, 2023, 2024]
    yidx = year_index(year, years)

    if count == 60:
        return yidx * 12 + (month - 1)
    if count == 20:
        if month not in {6, 7, 8, 9}:
            raise ValueError(
                f"{path} has 20 monsoon bands only; month {month} is unavailable. "
                "Use month 6,7,8,9 or a full-year manifest."
            )
        return yidx * 4 + [6, 7, 8, 9].index(month)
    if count == 1:
        return 0
    raise ValueError(f"Unsupported predictor band count for {path}: {count}")


def gt_band_index(path: str, year: int, month: int, target_period: str) -> int:
    with rasterio.open(path) as src:
        count = src.count

    years = [2020, 2021, 2022, 2023, 2024]
    yidx = year_index(year, years)

    if count == 65:
        if target_period == "annual":
            return yidx * 13 + 12
        return yidx * 13 + (month - 1)
    if count == 25:
        periods = [6, 7, 8, 9, "annual"]
        period = "annual" if target_period == "annual" else month
        if period not in periods:
            raise ValueError(
                f"{path} has 25 monsoon/annual bands only; target {period} is unavailable."
            )
        return yidx * 5 + periods.index(period)
    raise ValueError(f"Unsupported GT band count for {path}: {count}")


def read_band(path: str, band_idx_zero_based: int) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(band_idx_zero_based + 1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def finite_limits(arr: np.ndarray, p_low: float, p_high: float) -> tuple[float | None, float | None]:
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return None, None
    low, high = np.nanpercentile(values, [p_low, p_high])
    if np.isclose(low, high):
        return float(np.nanmin(values)), float(np.nanmax(values))
    return float(low), float(high)


def sanitize_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def plot_state(row: dict[str, str], args: argparse.Namespace, p_low: float, p_high: float) -> Path:
    state = row["state"]
    month_name = MONTH_NAMES[args.month]
    panels: list[tuple[str, np.ndarray, str, bool]] = []

    for predictor in PREDICTORS:
        band_idx = predictor_band_index(row[predictor], predictor, args.year, args.month)
        arr = read_band(row[predictor], band_idx)
        panels.append((PREDICTOR_TITLES[predictor], arr, CMAPS[predictor], False))

    if args.include_gt:
        gt_idx = gt_band_index(row["gt"], args.year, args.month, args.target_period)
        gt_arr = read_band(row["gt"], gt_idx)
        gt_title = "GT Annual Class" if args.target_period == "annual" else f"GT {month_name} Class"
        panels.append((gt_title, gt_arr, "gt", True))

    cols = 4 if len(panels) > 4 else len(panels)
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.2 * rows), squeeze=False)
    fig.suptitle(
        f"{state}: {month_name} {args.year} Multimodal Raster Maps",
        fontsize=15,
        fontweight="bold",
    )

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (title, arr, cmap, is_gt) in zip(axes.ravel(), panels):
        ax.axis("off")
        if is_gt:
            image = ax.imshow(arr, cmap=GT_CMAP, norm=GT_NORM, interpolation="nearest")
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 1, 2, 3, 4])
            cbar.ax.set_yticklabels(GT_CLASS_NAMES, fontsize=7)
            cbar.set_label("Rainfall anomaly class", fontsize=8)
        else:
            vmin, vmax = finite_limits(arr, p_low, p_high)
            image = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
            predictor_key = next(
                key for key, predictor_title in PREDICTOR_TITLES.items() if predictor_title == title
            )
            cbar.set_label(PREDICTOR_UNITS[predictor_key], fontsize=8)
        ax.set_title(title, fontsize=10)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    period = "annual" if args.target_period == "annual" else month_name
    output_path = output_dir / f"{sanitize_filename(state)}_{args.year}_{period}_modalities.png"
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    if args.month < 1 or args.month > 12:
        raise ValueError("--month must be between 1 and 12")
    p_low, p_high = parse_percentiles(args.percentile_clip)
    rows = load_manifest(Path(args.manifest))

    if not args.all_states:
        requested = {state.strip() for state in args.states.split(",") if state.strip()}
        if not requested:
            raise ValueError("Pass --states 'State A,State B' or use --all-states")
        rows = [row for row in rows if row["state"] in requested]
        missing = requested - {row["state"] for row in rows}
        if missing:
            raise ValueError(f"States not found in manifest: {', '.join(sorted(missing))}")

    if args.max_states > 0:
        rows = rows[: args.max_states]

    outputs = []
    for row in rows:
        print(f"Plotting {row['state']}...")
        outputs.append(plot_state(row, args, p_low, p_high))

    print("\nWrote:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
