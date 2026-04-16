#!/usr/bin/env python3
"""
Extract monthly autoregressive forecasting patches by Indian climate region.

Formulation:
    input months:  m-3, m-2, m-1
    target month:  m
    target years:  user-selected, typically 2021-2024

Default dynamic predictors:
    precipitation, LST, NDVI, relative humidity, soil moisture, wind speed

Static/context predictor:
    elevation, lulc, or none

The output layout matches train_patch_baselines.py:
    X.npy       (N, C, patch_size, patch_size)
    y.npy       (N,)
    samples.csv metadata, including region, target month, and transition label
    config.json channel layout for Conv3D/ConvLSTM/Swin3D

Notes:
    - Full Jan-Dec autoregressive extraction requires 60-band predictor stacks
      and 65-band precipitation GT stacks.
    - Some legacy states in this project only have 20/25 monsoon bands; they are
      skipped by default and reported in config.json.
    - LULC is not available for all current states. Use --static-predictor
      elevation for a complete current run, or export/process LULC first.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


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

DEFAULT_DYNAMIC_MODALITIES = [
    "precipitation",
    "lst",
    "ndvi",
    "rh",
    "soil_moisture",
    "wind_speed",
]

MANIFEST_DYNAMIC_COLUMNS = ["lst", "ndvi", "rh", "soil_moisture", "wind_speed"]


@dataclass
class StateBundle:
    state: str
    region: str
    gt_path: Path
    raw_precip_path: Path
    predictor_paths: dict[str, Path]
    static_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract monthly autoregressive patches")
    parser.add_argument("--manifest", default="modeling_manifest.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--region",
        required=True,
        choices=sorted(REGION_GROUPS),
        help="Climate region to extract",
    )
    parser.add_argument(
        "--dynamic-modalities",
        default=",".join(DEFAULT_DYNAMIC_MODALITIES),
        help=(
            "Comma-separated dynamic modalities. Supported: precipitation,lst,ndvi,rh,"
            "soil_moisture,wind_speed"
        ),
    )
    parser.add_argument(
        "--static-predictor",
        default="elevation",
        choices=["elevation", "lulc", "none"],
        help="Static/context predictor. Use lulc only after LULC exists for all selected states.",
    )
    parser.add_argument(
        "--target-years",
        default="2021,2022,2023,2024",
        help="Comma-separated target years. Use 2021-2024 for all Jan-Dec targets with 2020-2024 data.",
    )
    parser.add_argument(
        "--target-months",
        default="1,2,3,4,5,6,7,8,9,10,11,12",
        help="Comma-separated target months.",
    )
    parser.add_argument("--input-window", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-per-state", type=int, default=0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument(
        "--skip-incomplete-states",
        action="store_true",
        help="Skip states without full 60/65-band monthly stacks instead of failing.",
    )
    parser.add_argument(
        "--allow-missing-lulc",
        action="store_true",
        help="Skip states missing LULC when --static-predictor lulc.",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_modalities(text: str) -> list[str]:
    aliases = {
        "precip": "precipitation",
        "rain": "precipitation",
        "rainfall": "precipitation",
        "humidity": "rh",
        "relative_humidity": "rh",
        "soil": "soil_moisture",
        "soilmoisture": "soil_moisture",
        "wind": "wind_speed",
        "windspeed": "wind_speed",
    }
    supported = set(DEFAULT_DYNAMIC_MODALITIES)
    modalities = []
    for token in text.split(","):
        key = aliases.get(token.strip().lower(), token.strip().lower())
        if not key:
            continue
        if key not in supported:
            raise ValueError(f"Unsupported dynamic modality: {key}")
        modalities.append(key)
    if not modalities:
        raise ValueError("At least one dynamic modality is required")
    return modalities


def month_shift(year: int, month: int, delta: int) -> tuple[int, int]:
    absolute = year * 12 + (month - 1) + delta
    return absolute // 12, absolute % 12 + 1


def monthly_predictor_band(year: int, month: int, years: list[int]) -> int | None:
    if year not in years:
        return None
    return years.index(year) * 12 + (month - 1)


def monthly_gt_band(year: int, month: int, years: list[int]) -> int | None:
    if year not in years:
        return None
    return years.index(year) * 13 + (month - 1)


def infer_raw_precip_path(gt_path: Path) -> Path:
    raw = Path(str(gt_path).replace("_Y_Precipitation_GT_geotif.tif", "_Y_Precipitation_CHIRPS.tif"))
    return raw


def infer_lulc_path(state: str, gt_path: Path) -> Path:
    state_dir = gt_path.parent
    return state_dir / f"{state}_X_LULC_masked.tif"


def read_manifest(path: Path, region: str, static_predictor: str, allow_missing_lulc: bool) -> list[StateBundle]:
    selected_states = set(REGION_GROUPS[region])
    bundles: list[StateBundle] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"state", "gt", "elevation", *MANIFEST_DYNAMIC_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            state = row["state"].strip()
            if state not in selected_states:
                continue

            gt_path = Path(row["gt"].strip())
            raw_precip_path = infer_raw_precip_path(gt_path)
            predictor_paths = {col: Path(row[col].strip()) for col in MANIFEST_DYNAMIC_COLUMNS}

            static_path: Path | None
            if static_predictor == "none":
                static_path = None
            elif static_predictor == "elevation":
                static_path = Path(row["elevation"].strip())
            else:
                static_path = infer_lulc_path(state, gt_path)
                if not static_path.exists() and allow_missing_lulc:
                    continue

            bundles.append(
                StateBundle(
                    state=state,
                    region=region,
                    gt_path=gt_path,
                    raw_precip_path=raw_precip_path,
                    predictor_paths=predictor_paths,
                    static_path=static_path,
                )
            )
    return bundles


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(), {
            "count": src.count,
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs.to_string() if src.crs else None,
            "nodata": src.nodata,
        }


def ensure_same_grid(reference: dict, other: dict, path: Path) -> None:
    for key in ("height", "width", "transform", "crs"):
        if reference[key] != other[key]:
            raise ValueError(f"Grid mismatch for {path}: {key} differs")


def collapse_static(data: np.ndarray, name: str) -> np.ndarray:
    if data.shape[0] == 1:
        return data[:1].astype(np.float32)
    reference = data[0]
    if np.allclose(data, reference[None, ...], equal_nan=True):
        return data[:1].astype(np.float32)
    print(f"Warning: static predictor {name} has {data.shape[0]} bands; using first band only.")
    return data[:1].astype(np.float32)


def season_phase(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "pre_monsoon"
    if month == 6:
        return "monsoon_onset"
    if month in {7, 8}:
        return "monsoon_peak"
    if month == 9:
        return "monsoon_retreat"
    return "post_monsoon"


def transition_label(target_month: int) -> str:
    if target_month in {3, 4, 5}:
        return "winter_to_pre_monsoon"
    if target_month == 6:
        return "pre_monsoon_to_monsoon_onset"
    if target_month in {7, 8}:
        return "monsoon_onset_to_peak"
    if target_month in {10, 11}:
        return "monsoon_retreat_to_post_monsoon"
    if target_month == 9:
        return "monsoon_peak_to_retreat"
    return "post_monsoon_winter_context"


def build_channel_names(
    dynamic_modalities: list[str],
    static_predictor: str,
    input_window: int,
) -> list[str]:
    names = []
    if static_predictor != "none":
        names.append(f"{static_predictor}_static")
    for modality in dynamic_modalities:
        for offset in range(input_window, 0, -1):
            names.append(f"{modality}_tminus{offset}")
    return names


def extract_state(
    bundle: StateBundle,
    dynamic_modalities: list[str],
    static_predictor: str,
    target_years: list[int],
    target_months: list[int],
    source_years: list[int],
    input_window: int,
    horizon: int,
    patch_size: int,
    stride: int,
    max_per_state: int,
    min_valid_fraction: float,
    fill_value: float,
    skip_incomplete_states: bool,
) -> tuple[list[np.ndarray], list[int], list[dict], str | None]:
    gt_data, gt_meta = read_raster(bundle.gt_path)
    if gt_data.shape[0] != 65:
        message = f"{bundle.state} GT has {gt_data.shape[0]} bands, expected 65 for Jan-Dec monthly AR"
        if skip_incomplete_states:
            return [], [], [], message
        raise ValueError(message)
    gt_data = gt_data.astype(np.int16)

    raw_precip = None
    predictor_arrays: dict[str, np.ndarray] = {}
    if "precipitation" in dynamic_modalities:
        if not bundle.raw_precip_path.exists():
            raise FileNotFoundError(bundle.raw_precip_path)
        raw_precip, raw_meta = read_raster(bundle.raw_precip_path)
        ensure_same_grid(gt_meta, raw_meta, bundle.raw_precip_path)
        if raw_precip.shape[0] != 65:
            message = f"{bundle.state} raw precipitation has {raw_precip.shape[0]} bands, expected 65"
            if skip_incomplete_states:
                return [], [], [], message
            raise ValueError(message)
        raw_precip = raw_precip.astype(np.float32)

    for modality in dynamic_modalities:
        if modality == "precipitation":
            continue
        path = bundle.predictor_paths[modality]
        data, meta = read_raster(path)
        ensure_same_grid(gt_meta, meta, path)
        if data.shape[0] != 60:
            message = f"{bundle.state} {modality} has {data.shape[0]} bands, expected 60"
            if skip_incomplete_states:
                return [], [], [], message
            raise ValueError(message)
        predictor_arrays[modality] = data.astype(np.float32)

    static_array = None
    if static_predictor != "none":
        if bundle.static_path is None or not bundle.static_path.exists():
            raise FileNotFoundError(f"{bundle.state} static predictor missing: {bundle.static_path}")
        static_data, static_meta = read_raster(bundle.static_path)
        ensure_same_grid(gt_meta, static_meta, bundle.static_path)
        static_array = collapse_static(static_data, static_predictor)

    radius = patch_size // 2
    if patch_size % 2 == 0:
        raise ValueError("--patch-size must be odd")

    X: list[np.ndarray] = []
    y_labels: list[int] = []
    rows: list[dict] = []
    accepted = 0

    valid_targets: list[tuple[int, int, list[tuple[int, int]]]] = []
    for target_year in target_years:
        for target_month in target_months:
            input_months = [
                month_shift(target_year, target_month, -horizon - lag)
                for lag in range(input_window - 1, -1, -1)
            ]
            if all(input_year in source_years for input_year, _ in input_months):
                valid_targets.append((target_year, target_month, input_months))

    for row in range(radius, gt_meta["height"] - radius, stride):
        for col in range(radius, gt_meta["width"] - radius, stride):
            for target_year, target_month, input_months in valid_targets:
                target_band = monthly_gt_band(target_year, target_month, source_years)
                if target_band is None:
                    continue
                label = int(gt_data[target_band, row, col])
                if label < 0:
                    continue

                channels = []
                if static_array is not None:
                    channels.append(static_array)

                for modality in dynamic_modalities:
                    modality_channels = []
                    for input_year, input_month in input_months:
                        if modality == "precipitation":
                            assert raw_precip is not None
                            band = monthly_gt_band(input_year, input_month, source_years)
                            if band is None:
                                raise ValueError("Internal error: precipitation input band missing")
                            modality_channels.append(raw_precip[band : band + 1])
                        else:
                            band = monthly_predictor_band(input_year, input_month, source_years)
                            if band is None:
                                raise ValueError("Internal error: predictor input band missing")
                            modality_channels.append(predictor_arrays[modality][band : band + 1])
                    channels.append(np.concatenate(modality_channels, axis=0))

                patch_stack = np.concatenate(channels, axis=0)[
                    :, row - radius : row + radius + 1, col - radius : col + radius + 1
                ]
                finite = np.isfinite(patch_stack)
                valid_fraction = float(finite.mean())
                if valid_fraction < min_valid_fraction:
                    continue
                if not finite.all():
                    patch_stack = np.where(finite, patch_stack, fill_value)

                X.append(patch_stack.astype(np.float32))
                y_labels.append(label)
                rows.append(
                    {
                        "state": bundle.state,
                        "region": bundle.region,
                        "region_label": REGION_LABELS[bundle.region],
                        "row": row,
                        "col": col,
                        "label": label,
                        "year": target_year,
                        "target_year": target_year,
                        "target_month": target_month,
                        "target_name": MONTH_NUM_TO_NAME[target_month],
                        "target_period": "month",
                        "target_band": target_band + 1,
                        "input_year": ",".join(str(y) for y, _ in input_months),
                        "input_months": ",".join(str(m) for _, m in input_months),
                        "input_window": input_window,
                        "horizon": horizon,
                        "season_phase": season_phase(target_month),
                        "transition_label": transition_label(target_month),
                        "valid_fraction": valid_fraction,
                    }
                )
                accepted += 1
                if max_per_state and accepted >= max_per_state:
                    return X, y_labels, rows, None

    return X, y_labels, rows, None


def main() -> None:
    args = parse_args()
    dynamic_modalities = parse_modalities(args.dynamic_modalities)
    target_years = parse_int_list(args.target_years)
    target_months = parse_int_list(args.target_months)
    source_years = [2020, 2021, 2022, 2023, 2024]

    if args.input_window < 1:
        raise ValueError("--input-window must be >= 1")
    if args.horizon != 1:
        raise ValueError("This script currently supports --horizon 1")
    for month in target_months:
        if month < 1 or month > 12:
            raise ValueError(f"Invalid target month: {month}")

    bundles = read_manifest(
        Path(args.manifest),
        region=args.region,
        static_predictor=args.static_predictor,
        allow_missing_lulc=args.allow_missing_lulc,
    )
    if not bundles:
        raise SystemExit(f"No states found for region {args.region} in {args.manifest}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_all: list[np.ndarray] = []
    y_all: list[int] = []
    sample_rows: list[dict] = []
    state_counts: dict[str, int] = {}
    skipped_states: dict[str, str] = {}

    channel_names = build_channel_names(dynamic_modalities, args.static_predictor, args.input_window)

    for bundle in bundles:
        print(f"Extracting {args.region}: {bundle.state}")
        X_state, y_state, rows_state, skipped_reason = extract_state(
            bundle=bundle,
            dynamic_modalities=dynamic_modalities,
            static_predictor=args.static_predictor,
            target_years=target_years,
            target_months=target_months,
            source_years=source_years,
            input_window=args.input_window,
            horizon=args.horizon,
            patch_size=args.patch_size,
            stride=args.stride,
            max_per_state=args.max_per_state,
            min_valid_fraction=args.min_valid_fraction,
            fill_value=args.fill_value,
            skip_incomplete_states=args.skip_incomplete_states,
        )
        if skipped_reason:
            print(f"  skipped: {skipped_reason}")
            skipped_states[bundle.state] = skipped_reason
            continue
        print(f"  accepted {len(rows_state)} samples")
        X_all.extend(X_state)
        y_all.extend(y_state)
        sample_rows.extend(rows_state)
        state_counts[bundle.state] = len(rows_state)

    if not X_all:
        raise SystemExit("No samples extracted. Check region, full-year data availability, and LULC availability.")

    X = np.stack(X_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)

    fieldnames = [
        "sample_id",
        "state",
        "region",
        "region_label",
        "row",
        "col",
        "label",
        "year",
        "target_year",
        "target_month",
        "target_name",
        "target_period",
        "target_band",
        "input_year",
        "input_months",
        "input_window",
        "horizon",
        "season_phase",
        "transition_label",
        "valid_fraction",
    ]
    with (output_dir / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id, row in enumerate(sample_rows):
            writer.writerow({"sample_id": sample_id, **row})

    config = {
        "manifest": args.manifest,
        "sample_mode": "monthly_autoregressive",
        "region": args.region,
        "region_label": REGION_LABELS[args.region],
        "region_states_requested": REGION_GROUPS[args.region],
        "state_counts": state_counts,
        "skipped_states": skipped_states,
        "dynamic_modalities": dynamic_modalities,
        "static_predictor": args.static_predictor,
        "static_predictor_columns": [] if args.static_predictor == "none" else [args.static_predictor],
        "dynamic_predictor_columns": dynamic_modalities,
        "predictor_columns": ([] if args.static_predictor == "none" else [args.static_predictor]) + dynamic_modalities,
        "channel_names": channel_names,
        "static_channel_count": 0 if args.static_predictor == "none" else 1,
        "dynamic_channel_count": len(dynamic_modalities),
        "time_steps": args.input_window,
        "input_window": args.input_window,
        "horizon": args.horizon,
        "target_years": target_years,
        "target_months": target_months,
        "source_years": source_years,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "max_per_state": args.max_per_state,
        "min_valid_fraction": args.min_valid_fraction,
        "fill_value": args.fill_value,
        "num_samples": int(X.shape[0]),
        "num_channels": int(X.shape[1]),
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved {X.shape[0]} samples to {output_dir}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    if skipped_states:
        print("Skipped states:")
        for state, reason in skipped_states.items():
            print(f"  - {state}: {reason}")


if __name__ == "__main__":
    main()

