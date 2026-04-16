#!/usr/bin/env python3
"""
Extract trainable patches from the processed state rasters listed in modeling_manifest.csv.

Output layout:
- X.npy: float32 array of shape (N, C, patch_size, patch_size)
- y.npy: int64 array of shape (N,)
- samples.csv: per-sample metadata
- config.json: extraction settings and channel layout

The extractor uses the center pixel of the GT patch as the label and rejects any
sample whose center label is nodata (-1) or whose predictor patch contains NaNs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

import numpy as np
import rasterio


PREDICTOR_COLUMNS = [
    "elevation",
    "lst",
    "ndvi",
    "rh",
    "soil_moisture",
    "wind_speed",
]

STATIC_PREDICTOR_COLUMNS = ["elevation"]
DYNAMIC_PREDICTOR_COLUMNS = [col for col in PREDICTOR_COLUMNS if col not in STATIC_PREDICTOR_COLUMNS]
MONTH_NAME_TO_NUM = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
MONTH_NUM_TO_NAME = {
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


@dataclass
class RasterBundle:
    state: str
    gt_path: str
    predictor_paths: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract trainable raster patches")
    parser.add_argument(
        "--manifest",
        default="modeling_manifest.csv",
        help="CSV with state, gt, and predictor raster paths",
    )
    parser.add_argument(
        "--output-dir",
        default="patch_dataset",
        help="Directory for extracted arrays and metadata",
    )
    parser.add_argument(
        "--sample-mode",
        default="year_forecast",
        choices=["legacy_multiyear", "year_forecast", "year_sequence_forecast"],
        help=(
            "How samples are constructed. "
            "'year_forecast' creates one sample per spatial patch and year using only antecedent "
            "months from that year. 'year_sequence_forecast' creates one sample per target year "
            "using the same months across multiple previous years. 'legacy_multiyear' keeps the "
            "original pooled multiyear behavior."
        ),
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=15,
        help="Odd-valued square patch size",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Sliding-window stride in pixels",
    )
    parser.add_argument(
        "--max-per-state",
        type=int,
        default=0,
        help="Optional cap on accepted samples per state; 0 means no cap",
    )
    parser.add_argument(
        "--bands",
        default="all",
        help=(
            "Band selection for monthly predictors and GT. "
            "'all' keeps all bands, or pass a comma-separated 1-based list such as '1,2,3'."
        ),
    )
    parser.add_argument(
        "--schema",
        default="monsoon_5yr",
        choices=["auto", "monsoon_5yr"],
        help=(
            "Band harmonization scheme. "
            "'monsoon_5yr' maps mixed legacy/full stacks into a common 5-year monsoon schema."
        ),
    )
    parser.add_argument(
        "--years",
        default="2020,2021,2022,2023,2024",
        help="Comma-separated years to include for year_forecast mode",
    )
    parser.add_argument(
        "--input-months",
        default="6,7,8",
        help="Comma-separated antecedent months for year_forecast mode, e.g. '6,7,8'",
    )
    parser.add_argument(
        "--target-period",
        default="month",
        choices=["month", "annual"],
        help="Target type for year_forecast mode: a specific month or the annual total category",
    )
    parser.add_argument(
        "--target-month",
        type=int,
        default=9,
        help="Target month for year_forecast mode when --target-period=month, e.g. 9 for September",
    )
    parser.add_argument(
        "--target-year-offset",
        type=int,
        default=0,
        help=(
            "Year offset between inputs and target in year_forecast mode. "
            "Use 0 for same-year prediction, 1 for year t -> year t+1 forecasting."
        ),
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=1,
        help=(
            "Number of previous input years for year_sequence_forecast. "
            "For example, --history-years 4 and --target-year-offset 1 uses "
            "years t-4..t-1 to predict target year t."
        ),
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.7,
        help="Minimum fraction of finite predictor values required to keep a patch",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=0.0,
        help="Value used to replace remaining NaNs in accepted predictor patches",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def validate_months(months: list[int]) -> list[int]:
    cleaned = []
    for month in months:
        if month < 1 or month > 12:
            raise ValueError(f"Month must be between 1 and 12, got {month}")
        cleaned.append(month)
    return cleaned


def load_manifest(path: str) -> list[RasterBundle]:
    bundles = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"state", "gt", *PREDICTOR_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            state = row["state"].strip()
            bundles.append(
                RasterBundle(
                    state=state,
                    gt_path=row["gt"].strip(),
                    predictor_paths={col: row[col].strip() for col in PREDICTOR_COLUMNS},
                )
            )
    return bundles


def parse_band_selection(text: str, band_count: int) -> list[int]:
    if text == "all":
        return list(range(band_count))

    selected = []
    for token in text.split(","):
        idx = int(token.strip()) - 1
        if idx < 0 or idx >= band_count:
            raise ValueError(f"Band index out of range: {idx + 1}")
        selected.append(idx)
    if not selected:
        raise ValueError("At least one band must be selected")
    return selected


def monsoon_predictor_indices(band_count: int, predictor_name: str) -> list[int]:
    if predictor_name == "elevation":
        return [0]
    if band_count == 20:
        return list(range(20))
    if band_count == 60:
        indices = []
        for year_offset in range(0, 60, 12):
            indices.extend([year_offset + 5, year_offset + 6, year_offset + 7, year_offset + 8])
        return indices
    raise ValueError(f"Unsupported predictor band count for monsoon_5yr: {predictor_name}={band_count}")


def monsoon_gt_indices(band_count: int) -> list[int]:
    if band_count == 25:
        return list(range(25))
    if band_count == 65:
        indices = []
        for year_offset in range(0, 65, 13):
            indices.extend([year_offset + 5, year_offset + 6, year_offset + 7, year_offset + 8, year_offset + 12])
        return indices
    raise ValueError(f"Unsupported GT band count for monsoon_5yr: {band_count}")


def choose_predictor_indices(schema: str, text: str, predictor_name: str, band_count: int) -> list[int]:
    if schema == "monsoon_5yr":
        return monsoon_predictor_indices(band_count, predictor_name)
    return parse_band_selection(text, band_count)


def choose_gt_indices(schema: str, text: str, band_count: int) -> list[int]:
    if schema == "monsoon_5yr":
        return monsoon_gt_indices(band_count)
    return parse_band_selection(text, band_count)


def normalized_channel_names(schema: str, predictor_name: str, selected: list[int]) -> list[str]:
    if schema == "monsoon_5yr":
        return [f"{predictor_name}_band_{i+1}" for i in range(len(selected))]
    return [f"{predictor_name}_band_{i+1}" for i in selected]


def predictor_band_lookup(schema: str, predictor_name: str, band_count: int, years: list[int]) -> dict[int, dict[int, int]]:
    if predictor_name == "elevation":
        return {}

    if schema != "monsoon_5yr":
        raise ValueError("year_forecast currently supports only --schema monsoon_5yr")

    if band_count == 60:
        months_per_year = list(range(1, 13))
        bands_per_year = 12
    elif band_count == 20:
        months_per_year = [6, 7, 8, 9]
        bands_per_year = 4
    else:
        raise ValueError(f"Unsupported predictor band count for year_forecast: {predictor_name}={band_count}")

    lookup: dict[int, dict[int, int]] = {}
    for year_idx, year in enumerate(years):
        lookup[year] = {}
        base = year_idx * bands_per_year
        for month_idx, month in enumerate(months_per_year):
            lookup[year][month] = base + month_idx
    return lookup


def gt_band_lookup(schema: str, band_count: int, years: list[int]) -> dict[int, dict[int | str, int]]:
    if schema != "monsoon_5yr":
        raise ValueError("year_forecast currently supports only --schema monsoon_5yr")

    if band_count == 65:
        periods: list[int | str] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "Annual"]
        bands_per_year = 13
    elif band_count == 25:
        periods = [6, 7, 8, 9, "Annual"]
        bands_per_year = 5
    else:
        raise ValueError(f"Unsupported GT band count for year_forecast: {band_count}")

    lookup: dict[int, dict[int | str, int]] = {}
    for year_idx, year in enumerate(years):
        lookup[year] = {}
        base = year_idx * bands_per_year
        for period_idx, period in enumerate(periods):
            lookup[year][period] = base + period_idx
    return lookup


def build_year_forecast_channel_names(input_months: list[int]) -> list[str]:
    channel_names = []
    for predictor_name in STATIC_PREDICTOR_COLUMNS:
        channel_names.append(f"{predictor_name}_static")
    for predictor_name in DYNAMIC_PREDICTOR_COLUMNS:
        for month in input_months:
            channel_names.append(f"{predictor_name}_{MONTH_NUM_TO_NAME[month]}")
    return channel_names


def build_year_sequence_channel_names(input_months: list[int], history_years: int) -> list[str]:
    channel_names = []
    for predictor_name in STATIC_PREDICTOR_COLUMNS:
        channel_names.append(f"{predictor_name}_static")
    for predictor_name in DYNAMIC_PREDICTOR_COLUMNS:
        for history_idx in range(history_years):
            for month in input_months:
                channel_names.append(f"{predictor_name}_yminus{history_years - history_idx}_{MONTH_NUM_TO_NAME[month]}")
    return channel_names


def format_target_name(target_period: int | str) -> str:
    if target_period == "Annual":
        return "Annual"
    return MONTH_NUM_TO_NAME[int(target_period)]


def collapse_static_predictor(data: np.ndarray, predictor_name: str) -> np.ndarray:
    if data.shape[0] == 1:
        return data

    reference = data[0]
    if np.allclose(data, reference[None, ...], equal_nan=True):
        return data[:1]

    print(
        f"Warning: static predictor {predictor_name} has {data.shape[0]} bands; "
        "using the first band only."
    )
    return data[:1]


def read_raster(path: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(), {
            "count": src.count,
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs.to_string() if src.crs else None,
            "nodata": src.nodata,
        }


def ensure_same_grid(reference_meta: dict, other_meta: dict, path: str) -> None:
    keys = ("height", "width", "transform", "crs")
    for key in keys:
        if reference_meta[key] != other_meta[key]:
            raise ValueError(f"Grid mismatch for {path}: {key} differs")


def extract_from_bundle(
    bundle: RasterBundle,
    patch_size: int,
    stride: int,
    max_per_state: int,
    band_selection_text: str,
    schema: str,
    sample_mode: str,
    years: list[int],
    input_months: list[int],
    target_period: int | str,
    target_year_offset: int,
    history_years: int,
    min_valid_fraction: float,
    fill_value: float,
) -> tuple[list[np.ndarray], list[int], list[dict], list[str]]:
    gt_data, gt_meta = read_raster(bundle.gt_path)
    reference_meta = gt_meta

    if sample_mode == "legacy_multiyear":
        gt_band_indices = choose_gt_indices(schema, band_selection_text, gt_data.shape[0])
        gt_data = gt_data[gt_band_indices].astype(np.int16)

        predictor_arrays = []
        channel_names = []

        for predictor_name in PREDICTOR_COLUMNS:
            path = bundle.predictor_paths[predictor_name]
            data, meta = read_raster(path)
            ensure_same_grid(reference_meta, meta, path)
            selected = choose_predictor_indices(schema, band_selection_text, predictor_name, data.shape[0])
            data = data[selected].astype(np.float32)
            predictor_arrays.append(data)
            channel_names.extend(normalized_channel_names(schema, predictor_name, selected))

        predictors = np.concatenate(predictor_arrays, axis=0)
        predictor_cube = predictors
        predictor_by_year: dict[int, np.ndarray] | None = None
        gt_lookup: dict[int, dict[int | str, int]] | None = None
    else:
        gt_data = gt_data.astype(np.int16)
        gt_lookup = gt_band_lookup(schema, gt_data.shape[0], years)
        predictor_by_year = {}
        if sample_mode == "year_sequence_forecast":
            channel_names = build_year_sequence_channel_names(input_months, history_years)
        else:
            channel_names = build_year_forecast_channel_names(input_months)

        static_arrays: dict[str, np.ndarray] = {}
        dynamic_arrays: dict[str, tuple[np.ndarray, dict[int, dict[int, int]]]] = {}

        for predictor_name in PREDICTOR_COLUMNS:
            path = bundle.predictor_paths[predictor_name]
            data, meta = read_raster(path)
            ensure_same_grid(reference_meta, meta, path)
            data = data.astype(np.float32)
            if predictor_name in STATIC_PREDICTOR_COLUMNS:
                static_arrays[predictor_name] = collapse_static_predictor(data, predictor_name)
            else:
                dynamic_arrays[predictor_name] = (
                    data,
                    predictor_band_lookup(schema, predictor_name, data.shape[0], years),
                )

        for year in years:
            channels_for_year = []
            for predictor_name in STATIC_PREDICTOR_COLUMNS:
                channels_for_year.append(static_arrays[predictor_name])
            for predictor_name in DYNAMIC_PREDICTOR_COLUMNS:
                data, lookup = dynamic_arrays[predictor_name]
                month_indices = []
                for month in input_months:
                    if month not in lookup[year]:
                        raise ValueError(
                            f"Predictor {predictor_name} does not contain month {month} for year {year}"
                        )
                    month_indices.append(lookup[year][month])
                channels_for_year.append(data[month_indices])
            predictor_by_year[year] = np.concatenate(channels_for_year, axis=0)
        sequence_predictor_by_target: dict[int, tuple[np.ndarray, str]] = {}
        if sample_mode == "year_sequence_forecast":
            for target_year in years:
                final_input_year = target_year - target_year_offset
                input_years = list(range(final_input_year - history_years + 1, final_input_year + 1))
                if target_year not in gt_lookup or any(input_year not in predictor_by_year for input_year in input_years):
                    continue

                sequence_channels = []
                for predictor_name in STATIC_PREDICTOR_COLUMNS:
                    sequence_channels.append(static_arrays[predictor_name])
                for predictor_idx, predictor_name in enumerate(DYNAMIC_PREDICTOR_COLUMNS):
                    start = len(STATIC_PREDICTOR_COLUMNS)
                    block_start = start + predictor_idx * len(input_months)
                    block_end = block_start + len(input_months)
                    for input_year in input_years:
                        sequence_channels.append(predictor_by_year[input_year][block_start:block_end])
                input_year_for_metadata = ",".join(str(input_year) for input_year in input_years)
                sequence_predictor_by_target[target_year] = (
                    np.concatenate(sequence_channels, axis=0),
                    input_year_for_metadata,
                )
        predictor_cube = None

    radius = patch_size // 2
    if patch_size % 2 == 0:
        raise ValueError("--patch-size must be odd")

    X_patches: list[np.ndarray] = []
    y_labels: list[int] = []
    samples: list[dict] = []

    accepted = 0

    for y in range(radius, reference_meta["height"] - radius, stride):
        for x in range(radius, reference_meta["width"] - radius, stride):
            if sample_mode == "legacy_multiyear":
                gt_patch = gt_data[:, y - radius : y + radius + 1, x - radius : x + radius + 1]
                predictor_patch = predictor_cube[:, y - radius : y + radius + 1, x - radius : x + radius + 1]

                finite_mask = np.isfinite(predictor_patch)
                valid_fraction = float(finite_mask.mean())
                if valid_fraction < min_valid_fraction:
                    continue

                if not finite_mask.all():
                    predictor_patch = np.where(finite_mask, predictor_patch, fill_value)

                band_count_gt = gt_data.shape[0]
                for target_band_idx in range(band_count_gt):
                    center_label = int(gt_patch[target_band_idx, radius, radius])
                    if center_label < 0:
                        continue

                    X_patches.append(predictor_patch)
                    y_labels.append(center_label)
                    samples.append(
                        {
                            "state": bundle.state,
                            "row": y,
                            "col": x,
                            "label": center_label,
                            "target_band": target_band_idx + 1,
                            "year": "",
                            "input_year": "",
                            "target_year": "",
                            "input_months": "",
                            "target_month": "",
                            "target_period": "",
                            "target_name": "",
                            "valid_fraction": valid_fraction,
                        }
                    )
                    accepted += 1

                    if max_per_state and accepted >= max_per_state:
                        return X_patches, y_labels, samples, channel_names
            else:
                assert predictor_by_year is not None
                assert gt_lookup is not None
                if sample_mode == "year_sequence_forecast":
                    target_years = list(sequence_predictor_by_target)
                else:
                    target_years = [input_year + target_year_offset for input_year in years]

                for target_year in target_years:
                    if target_year not in gt_lookup:
                        continue
                    if target_period not in gt_lookup[target_year]:
                        raise ValueError(
                            f"GT does not contain target period {target_period} for year {target_year}"
                        )

                    if sample_mode == "year_sequence_forecast":
                        predictor_stack, input_year_for_metadata = sequence_predictor_by_target[target_year]
                    else:
                        input_year = target_year - target_year_offset
                        if input_year not in predictor_by_year:
                            continue
                        predictor_stack = predictor_by_year[input_year]
                        input_year_for_metadata = input_year

                    predictor_patch = predictor_stack[:, y - radius : y + radius + 1, x - radius : x + radius + 1]
                    finite_mask = np.isfinite(predictor_patch)
                    valid_fraction = float(finite_mask.mean())
                    if valid_fraction < min_valid_fraction:
                        continue

                    if not finite_mask.all():
                        predictor_patch = np.where(finite_mask, predictor_patch, fill_value)

                    target_band_idx = gt_lookup[target_year][target_period]
                    center_label = int(gt_data[target_band_idx, y, x])
                    if center_label < 0:
                        continue

                    X_patches.append(predictor_patch)
                    y_labels.append(center_label)
                    samples.append(
                        {
                            "state": bundle.state,
                            "row": y,
                            "col": x,
                            "label": center_label,
                            "target_band": target_band_idx + 1,
                            "year": target_year,
                            "input_year": input_year_for_metadata,
                            "target_year": target_year,
                            "input_months": ",".join(str(month) for month in input_months),
                            "target_month": "" if target_period == "Annual" else int(target_period),
                            "target_period": target_period,
                            "target_name": format_target_name(target_period),
                            "valid_fraction": valid_fraction,
                        }
                    )
                    accepted += 1

                    if max_per_state and accepted >= max_per_state:
                        return X_patches, y_labels, samples, channel_names

    return X_patches, y_labels, samples, channel_names


def main() -> None:
    args = parse_args()
    bundles = load_manifest(args.manifest)
    years = parse_int_list(args.years)
    input_months = validate_months(parse_int_list(args.input_months))
    if args.history_years < 1:
        raise ValueError("--history-years must be >= 1")
    if args.target_period == "annual":
        target_period: int | str = "Annual"
    else:
        target_period = validate_months([args.target_month])[0]
        if args.target_year_offset == 0 and target_period in input_months:
            raise ValueError("--target-month must not be one of --input-months for same-year forecasting")

    os.makedirs(args.output_dir, exist_ok=True)

    X_all: list[np.ndarray] = []
    y_all: list[int] = []
    sample_rows: list[dict] = []
    channel_names: list[str] | None = None
    state_counts = {}

    for bundle in bundles:
        print(f"Extracting patches for {bundle.state}...")
        X_state, y_state, samples_state, channels_state = extract_from_bundle(
            bundle=bundle,
            patch_size=args.patch_size,
            stride=args.stride,
            max_per_state=args.max_per_state,
            band_selection_text=args.bands,
            schema=args.schema,
            sample_mode=args.sample_mode,
            years=years,
            input_months=input_months,
            target_period=target_period,
            target_year_offset=args.target_year_offset,
            history_years=args.history_years,
            min_valid_fraction=args.min_valid_fraction,
            fill_value=args.fill_value,
        )
        if channel_names is None:
            channel_names = channels_state
        elif channel_names != channels_state:
            raise ValueError(f"Channel layout mismatch for {bundle.state}")

        X_all.extend(X_state)
        y_all.extend(y_state)
        sample_rows.extend(samples_state)
        state_counts[bundle.state] = len(samples_state)
        print(f"Accepted {len(samples_state)} patches for {bundle.state}")

    if not X_all:
        raise SystemExit("No patches were extracted. Check patch size, stride, and nodata coverage.")

    X = np.stack(X_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)

    np.save(os.path.join(args.output_dir, "X.npy"), X)
    np.save(os.path.join(args.output_dir, "y.npy"), y)

    samples_csv = os.path.join(args.output_dir, "samples.csv")
    with open(samples_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "state",
                "row",
                "col",
                "label",
                "target_band",
                "year",
                "input_year",
                "target_year",
                "input_months",
                "target_month",
                "target_period",
                "target_name",
                "valid_fraction",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(sample_rows):
            writer.writerow({"sample_id": idx, **row})

    config = {
        "manifest": args.manifest,
        "sample_mode": args.sample_mode,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "max_per_state": args.max_per_state,
        "bands": args.bands,
        "schema": args.schema,
        "years": years,
        "input_months": input_months,
        "target_period": target_period,
        "target_year_offset": args.target_year_offset,
        "history_years": args.history_years,
        "min_valid_fraction": args.min_valid_fraction,
        "fill_value": args.fill_value,
        "predictor_columns": PREDICTOR_COLUMNS,
        "static_predictor_columns": STATIC_PREDICTOR_COLUMNS,
        "dynamic_predictor_columns": DYNAMIC_PREDICTOR_COLUMNS,
        "channel_names": channel_names,
        "static_channel_count": len(STATIC_PREDICTOR_COLUMNS)
        if args.sample_mode in {"year_forecast", "year_sequence_forecast"}
        else 0,
        "dynamic_channel_count": len(DYNAMIC_PREDICTOR_COLUMNS)
        if args.sample_mode in {"year_forecast", "year_sequence_forecast"}
        else 0,
        "time_steps": len(input_months)
        if args.sample_mode == "year_forecast"
        else len(input_months) * args.history_years
        if args.sample_mode == "year_sequence_forecast"
        else 0,
        "month_steps": len(input_months) if args.sample_mode == "year_sequence_forecast" else 0,
        "year_steps": args.history_years if args.sample_mode == "year_sequence_forecast" else 0,
        "num_samples": int(X.shape[0]),
        "num_channels": int(X.shape[1]),
        "state_counts": state_counts,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved {X.shape[0]} samples to {args.output_dir}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    main()
