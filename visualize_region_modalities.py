#!/usr/bin/env python3
"""
Visualize a climate-region mosaic for a selected year/month.

The script reads the processed raster paths from modeling_manifest.csv and
plots the same variables used in the monthly autoregressive benchmark:
  - elevation
  - precipitation class/GT
  - LST, NDVI, relative humidity, soil moisture, wind speed

It does not use LULC.

Example:
    python visualize_region_modalities.py \
      --manifest modeling_manifest.csv \
      --region northwest_himalayan \
      --year 2024 \
      --month 4 \
      --output-dir region_modality_maps
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    raise SystemExit("geopandas is required for state boundary overlays") from exc

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for visualization") from exc

try:
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
except Exception as exc:  # pragma: no cover
    raise SystemExit("rasterio is required for reading GeoTIFFs") from exc


REGION_GROUPS = {
    "northwest_himalayan": [
        "Haryana",
        "Himachal Pradesh",
        "Punjab",
        "Rajasthan",
        "Uttarakhand",
        "Jammu and Kashmir",
    ],
    "central_monsoon_core": [
        "Bihar",
        "Chhattisgarh",
        "Jharkhand",
        "Madhya Pradesh",
        "Maharashtra",
        "Odisha",
        "Uttar Pradesh",
    ],
    "south_peninsular_deccan": [
        "Andhra Pradesh",
        "Goa",
        "Karnataka",
        "Kerala",
        "Tamil Nadu",
        "Telangana",
    ],
    "east_northeast_humid_orographic": [
        "Arunachal Pradesh",
        "Assam",
        "Manipur",
        "Meghalaya",
        "Mizoram",
        "Nagaland",
        "Sikkim",
        "Tripura",
        "West Bengal",
    ],
}

REGION_LABELS = {
    "northwest_himalayan": "Northwest-Himalayan Rainfall Regime",
    "central_monsoon_core": "Central Indian Monsoon Core Regime",
    "south_peninsular_deccan": "South Peninsular-Deccan Rainfall Regime",
    "east_northeast_humid_orographic": "East-Northeast Humid Orographic Regime",
}

PANEL_ORDER = [
    "elevation",
    "gt",
    "lst",
    "ndvi",
    "rh",
    "soil_moisture",
    "wind_speed",
]

PANEL_TITLES = {
    "elevation": "Elevation",
    "gt": "Rainfall Class",
    "lst": "Land Surface Temperature",
    "ndvi": "NDVI",
    "rh": "Relative Humidity",
    "soil_moisture": "Soil Moisture",
    "wind_speed": "Wind Speed",
}

PANEL_UNITS = {
    "elevation": "m",
    "gt": "class",
    "lst": "deg C",
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

GT_CLASS_NAMES = ["Scarcity", "Deficit", "Normal", "Excess", "Large Excess"]
GT_CMAP = ListedColormap(["#8c510a", "#d8b365", "#5ab4ac", "#2b83ba", "#762a83"])
GT_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], GT_CMAP.N)

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

FULL_YEARS = [2020, 2021, 2022, 2023, 2024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize regional multimodal raster mosaics")
    parser.add_argument("--manifest", default="modeling_manifest.csv")
    parser.add_argument("--region", required=True, choices=sorted(REGION_GROUPS))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=9)
    parser.add_argument("--output-dir", default="region_modality_maps")
    parser.add_argument(
        "--percentile-clip",
        default="2,98",
        help="Continuous color scale clipping, e.g. 2,98",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--states",
        default="",
        help="Optional comma-separated subset of states inside the region.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip states/files that are missing or incompatible instead of failing.",
    )
    parser.add_argument(
        "--show-state-labels",
        action="store_true",
        help="Overlay state labels at approximate raster centroids.",
    )
    parser.add_argument(
        "--show-boundaries",
        action="store_true",
        help="Overlay state boundary outlines on each panel.",
    )
    parser.add_argument(
        "--boundary-shapefile",
        default="",
        help="Optional GADM state boundary shapefile path. Defaults to the first GridData/*/SateMask/gadm41_IND_1.shp found.",
    )
    parser.add_argument("--boundary-linewidth", type=float, default=0.8)
    parser.add_argument("--boundary-color", default="black")
    parser.add_argument(
        "--fill-visual-nodata",
        action="store_true",
        help=(
            "Visualization-only nearest-neighbor fill for nodata holes inside "
            "state boundaries. Outside-state pixels remain transparent."
        ),
    )
    return parser.parse_args()


def parse_percentiles(text: str) -> tuple[float, float]:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if len(values) != 2:
        raise ValueError("--percentile-clip must contain two values, e.g. 2,98")
    return values[0], values[1]


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["state"].strip(): dict(row) for row in csv.DictReader(f)}


def find_boundary_shapefile(explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for path in sorted(Path("GridData").glob("*/SateMask/gadm41_IND_1.shp")):
        return path
    raise FileNotFoundError("Could not find GridData/*/SateMask/gadm41_IND_1.shp")


def load_boundaries(states: list[str], crs, shapefile: Path):
    gdf = gpd.read_file(shapefile)
    if "NAME_1" not in gdf.columns:
        raise ValueError(f"Boundary shapefile is missing NAME_1 column: {shapefile}")
    selected = gdf[gdf["NAME_1"].isin(states)].copy()
    if selected.empty:
        return selected
    if crs is not None and selected.crs != crs:
        selected = selected.to_crs(crs)
    return selected


def load_boundary_table(states: list[str], shapefile: Path):
    gdf = gpd.read_file(shapefile)
    if "NAME_1" not in gdf.columns:
        raise ValueError(f"Boundary shapefile is missing NAME_1 column: {shapefile}")
    return gdf[gdf["NAME_1"].isin(states)].copy()


def requested_states(args: argparse.Namespace) -> list[str]:
    region_states = REGION_GROUPS[args.region]
    if not args.states:
        return region_states
    requested = [token.strip() for token in args.states.split(",") if token.strip()]
    outside = sorted(set(requested) - set(region_states))
    if outside:
        raise ValueError(f"States are not in {args.region}: {', '.join(outside)}")
    return requested


def monthly_predictor_band(path: str, predictor: str, year: int, month: int) -> int:
    with rasterio.open(path) as src:
        count = src.count
    if predictor == "elevation":
        return 1
    if year not in FULL_YEARS:
        raise ValueError(f"Unsupported year {year}; expected one of {FULL_YEARS}")
    yidx = FULL_YEARS.index(year)
    if count == 60:
        return yidx * 12 + month
    if count == 20:
        if month not in {6, 7, 8, 9}:
            raise ValueError(f"{path} has only Jun-Sep bands; month {month} unavailable")
        return yidx * 4 + [6, 7, 8, 9].index(month) + 1
    if count == 1:
        return 1
    raise ValueError(f"Unsupported band count for {path}: {count}")


def monthly_gt_band(path: str, year: int, month: int) -> int:
    with rasterio.open(path) as src:
        count = src.count
    if year not in FULL_YEARS:
        raise ValueError(f"Unsupported year {year}; expected one of {FULL_YEARS}")
    yidx = FULL_YEARS.index(year)
    if count == 65:
        return yidx * 13 + month
    if count == 25:
        if month not in {6, 7, 8, 9}:
            raise ValueError(f"{path} has only Jun-Sep monthly GT; month {month} unavailable")
        return yidx * 5 + [6, 7, 8, 9].index(month) + 1
    raise ValueError(f"Unsupported GT band count for {path}: {count}")


def clean_array(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    arr = arr.astype(np.float32)
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def state_inside_mask(boundary_table, state: str, src) -> np.ndarray | None:
    if boundary_table is None or boundary_table.empty:
        return None
    selected = boundary_table[boundary_table["NAME_1"] == state]
    if selected.empty:
        return None
    if selected.crs is not None and src.crs is not None and selected.crs != src.crs:
        selected = selected.to_crs(src.crs)
    geom = selected.geometry.unary_union
    return geometry_mask(
        [geom],
        out_shape=(src.height, src.width),
        transform=src.transform,
        invert=True,
    )


def nearest_fill_inside_boundary(arr: np.ndarray, inside: np.ndarray | None) -> np.ndarray:
    """Fill visual nodata holes inside the state only; keep outside transparent."""
    if inside is None:
        return arr
    out = arr.copy()
    valid = np.isfinite(out) & inside
    missing_inside = (~np.isfinite(out)) & inside
    if not np.any(missing_inside) or not np.any(valid):
        return out

    # distance_transform_edt returns indices of the nearest zero. We pass
    # invalid=True and valid=False, so each invalid pixel receives nearest valid.
    invalid = ~valid
    _, indices = ndimage.distance_transform_edt(invalid, return_indices=True)
    out[missing_inside] = out[tuple(index[missing_inside] for index in indices)]
    out[~inside] = np.nan
    return out


def mosaic_band(
    rows: list[dict[str, str]],
    key: str,
    year: int,
    month: int,
    skip_missing: bool,
    boundary_table=None,
    fill_visual_nodata: bool = False,
):
    sources = []
    memfiles = []
    labels = []
    errors = []
    for row in rows:
        state = row["state"]
        path = row["gt"] if key == "gt" else row[key]
        try:
            band = monthly_gt_band(path, year, month) if key == "gt" else monthly_predictor_band(path, key, year, month)
            with rasterio.open(path) as src:
                if src.crs is None:
                    raise ValueError(f"{path} has no CRS")
                arr = clean_array(src.read(band), src.nodata)
                inside = state_inside_mask(boundary_table, state, src)
                if inside is not None:
                    arr[~inside] = np.nan
                if fill_visual_nodata:
                    arr = nearest_fill_inside_boundary(arr, inside)
                profile = src.profile.copy()
                profile.update(count=1, dtype="float32", nodata=np.nan)
                memfile = MemoryFile()
                dataset = memfile.open(**profile)
                dataset.write(arr.astype(np.float32), 1)
            memfiles.append(memfile)
            sources.append((dataset, 1))
            labels.append(state)
        except Exception as exc:
            errors.append(f"{state}: {exc}")
            if not skip_missing:
                for src, _ in sources:
                    src.close()
                for memfile in memfiles:
                    memfile.close()
                raise

    if not sources:
        raise ValueError(f"No usable rasters for panel {key}. Errors: {errors}")

    datasets = [src for src, _ in sources]
    try:
        mosaic, transform = merge(datasets, indexes=[1], method="first", nodata=np.nan)
        crs = datasets[0].crs
        centroids = []
        for state, (src, band) in zip(labels, sources):
            data = clean_array(src.read(band), src.nodata)
            rows_idx, cols_idx = np.where(np.isfinite(data))
            if rows_idx.size == 0:
                continue
            row_c = float(rows_idx.mean())
            col_c = float(cols_idx.mean())
            x, y = src.transform * (col_c, row_c)
            centroids.append((state, x, y))
    finally:
        for src, _ in sources:
            src.close()
        for memfile in memfiles:
            memfile.close()

    return clean_array(mosaic[0], np.nan), transform, crs, centroids, errors


def extent_from_transform(transform, width: int, height: int) -> tuple[float, float, float, float]:
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return left, right, bottom, top


def finite_limits(arr: np.ndarray, low: float, high: float) -> tuple[float | None, float | None]:
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return None, None
    vmin, vmax = np.nanpercentile(values, [low, high])
    if np.isclose(vmin, vmax):
        return float(np.nanmin(values)), float(np.nanmax(values))
    return float(vmin), float(vmax)


def add_state_labels(ax, centroids, transform, fontsize: int = 7) -> None:
    if not centroids:
        return
    for state, x, y in centroids:
        ax.text(
            x,
            y,
            state,
            fontsize=fontsize,
            ha="center",
            va="center",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1.5},
        )


def add_boundaries(ax, boundaries, color: str, linewidth: float) -> None:
    if boundaries is None or boundaries.empty:
        return
    boundaries.boundary.plot(ax=ax, color=color, linewidth=linewidth, alpha=0.9, zorder=5)


def main() -> None:
    args = parse_args()
    p_low, p_high = parse_percentiles(args.percentile_clip)
    manifest = load_manifest(Path(args.manifest))
    states = requested_states(args)
    rows = []
    missing_manifest = []
    for state in states:
        if state in manifest:
            rows.append(manifest[state])
        else:
            missing_manifest.append(state)
            if not args.skip_missing:
                raise ValueError(f"{state} is not present in {args.manifest}")

    if not rows:
        raise ValueError("No manifest rows selected")

    month_name = MONTH_NAMES[args.month]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    all_errors: list[str] = []
    mosaic_crs = None
    boundary_path = find_boundary_shapefile(args.boundary_shapefile) if (
        args.show_boundaries or args.fill_visual_nodata
    ) else None
    boundary_table = load_boundary_table([row["state"] for row in rows], boundary_path) if boundary_path else None
    for key in PANEL_ORDER:
        arr, transform, crs, centroids, errors = mosaic_band(
            rows,
            key,
            args.year,
            args.month,
            args.skip_missing,
            boundary_table=boundary_table,
            fill_visual_nodata=args.fill_visual_nodata,
        )
        if mosaic_crs is None:
            mosaic_crs = crs
        panels.append((key, arr, transform, centroids))
        all_errors.extend(errors)

    boundaries = None
    if args.show_boundaries:
        boundaries = load_boundaries([row["state"] for row in rows], mosaic_crs, boundary_path)

    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), constrained_layout=True)
    fig.suptitle(
        f"{REGION_LABELS[args.region]}: {month_name} {args.year} Multimodal Raster Mosaic",
        fontsize=17,
        fontweight="bold",
    )

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (key, arr, transform, centroids) in zip(axes.ravel(), panels):
        ax.axis("off")
        extent = extent_from_transform(transform, arr.shape[1], arr.shape[0])
        if key == "gt":
            image = ax.imshow(arr, cmap=GT_CMAP, norm=GT_NORM, interpolation="nearest", extent=extent)
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, ticks=[0, 1, 2, 3, 4])
            cbar.ax.set_yticklabels(GT_CLASS_NAMES, fontsize=8)
            cbar.set_label("Rainfall anomaly class", fontsize=8)
        else:
            vmin, vmax = finite_limits(arr, p_low, p_high)
            image = ax.imshow(
                arr,
                cmap=CMAPS[key],
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                extent=extent,
            )
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(PANEL_UNITS[key], fontsize=8)
        ax.set_title(PANEL_TITLES[key], fontsize=11)
        if args.show_boundaries:
            add_boundaries(ax, boundaries, args.boundary_color, args.boundary_linewidth)
        if args.show_state_labels:
            add_state_labels(ax, centroids, transform)

    axes.ravel()[-1].axis("off")
    note = f"States shown: {', '.join(row['state'] for row in rows)}"
    if missing_manifest:
        note += f" | Missing from manifest: {', '.join(missing_manifest)}"
    if all_errors:
        note += f" | Skipped panels/states: {len(all_errors)} issue(s)"
    fig.text(0.01, 0.01, note, fontsize=8)

    out_path = output_dir / f"{args.region}_{args.year}_{month_name.lower()}_modalities.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    if all_errors:
        error_path = output_dir / f"{args.region}_{args.year}_{month_name.lower()}_skipped.txt"
        error_path.write_text("\n".join(all_errors) + "\n")
        print(f"Wrote skipped-state notes: {error_path}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
