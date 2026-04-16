#!/usr/bin/env python3
"""
Submit serial GEE exports to repair missing full-monthly stacks and LULC.

This script does two things:

1. Full export for states whose current local files are incomplete for
   Jan-Dec monthly autoregressive training:
      - GT category source should be 65 bands after processing
      - dynamic predictors should be 60 bands after export
   Full export delegates to scripts/export_state_data.py and exports all 8 files.

2. LULC-only export for states that already have complete full-monthly stacks
   but are missing LULC locally.

The script only submits GEE tasks. It does not download from Google Drive and
does not overwrite local GridData files. After tasks complete, download/sync
files from Google Drive/GEE_Exports and run scripts/prepare_missing_states_from_exports.py
or scripts/process_state_data.py as appropriate.

Examples:
    # Dry run: see what would be submitted
    python3 scripts/export_repair_full_monthly_and_lulc.py \
      --project weather-analysis-461411 \
      --regions all \
      --dry-run

    # Submit all missing full exports + LULC-only exports serially
    python3 scripts/export_repair_full_monthly_and_lulc.py \
      --project weather-analysis-461411 \
      --regions all

    # Avoid duplicating states already submitted manually
    python3 scripts/export_repair_full_monthly_and_lulc.py \
      --project weather-analysis-461411 \
      --regions all \
      --exclude-full-states "Rajasthan,Himachal Pradesh,Bihar"
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

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

DYNAMIC_COLUMNS = ["lst", "ndvi", "rh", "soil_moisture", "wind_speed"]
EXPORT_YEARS = [2020, 2021, 2022, 2023, 2024]
EXPORT_MONTHS = list(range(1, 13))
MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair missing full monthly/LULC GEE exports")
    parser.add_argument("--project", required=True, help="Google Earth Engine project ID")
    parser.add_argument("--manifest", default="modeling_manifest.csv")
    parser.add_argument("--normals-csv", default="states_normals.csv")
    parser.add_argument(
        "--regions",
        default="all",
        help="Comma-separated region names or 'all'",
    )
    parser.add_argument(
        "--extra-full-states",
        default="",
        help="Comma-separated additional states to full-export even if not in manifest.",
    )
    parser.add_argument(
        "--exclude-full-states",
        default="",
        help="Comma-separated states to exclude from full exports, useful for tasks already submitted.",
    )
    parser.add_argument(
        "--exclude-lulc-states",
        default="",
        help="Comma-separated states to exclude from LULC-only exports.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Delay between export submissions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned exports without submitting GEE tasks.",
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def selected_states_for_regions(regions_text: str) -> list[str]:
    if regions_text.strip().lower() == "all":
        regions = list(REGION_GROUPS)
    else:
        regions = parse_csv_list(regions_text)
        unknown = [region for region in regions if region not in REGION_GROUPS]
        if unknown:
            raise ValueError(f"Unknown region(s): {', '.join(unknown)}")

    states = []
    seen = set()
    for region in regions:
        for state in REGION_GROUPS[region]:
            if state not in seen:
                states.append(state)
                seen.add(state)
    return states


def raster_count(path: str | Path) -> int | str:
    path = Path(path)
    if not path.exists():
        return "missing_file"
    with rasterio.open(path) as src:
        return int(src.count)


def infer_lulc_path(state: str, gt_path: str) -> Path:
    return Path(gt_path).parent / f"{state}_X_LULC_masked.tif"


def load_normals(path: Path) -> dict[str, str]:
    normals = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            normals[row["state"].strip()] = row["normals"].strip()
    return normals


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["state"].strip()] = {key: value.strip() for key, value in row.items()}
    return rows


def needs_full_export(row: dict[str, str]) -> tuple[bool, str]:
    issues = []
    if raster_count(row["gt"]) != 65:
        issues.append(f"gt={raster_count(row['gt'])}")
    for col in DYNAMIC_COLUMNS:
        count = raster_count(row[col])
        if count != 60:
            issues.append(f"{col}={count}")
    return bool(issues), "; ".join(issues)


def missing_lulc(row: dict[str, str]) -> bool:
    return raster_count(infer_lulc_path(row["state"], row["gt"])) == "missing_file"


def submit_full_export(state: str, normals: str, project: str, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        "scripts/export_state_data.py",
        "--state",
        state,
        "--normals",
        normals,
        "--project",
        project,
    ]
    print("FULL:", " ".join(f"'{x}'" if " " in x else x for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def submit_lulc_export(state: str, project: str, dry_run: bool) -> None:
    print(f"LULC-only: {state}")
    if dry_run:
        return

    import ee

    ee.Initialize(project=project)
    states = ee.FeatureCollection("FAO/GAUL/2015/level1")
    state_fc = states.filter(ee.Filter.eq("ADM1_NAME", state))

    bands = []
    for year in EXPORT_YEARS:
        for month, month_name in zip(EXPORT_MONTHS, MONTH_NAMES):
            start = ee.Date.fromYMD(year, month, 1)
            end = start.advance(1, "month")
            lulc = (
                ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                .filterDate(start, end)
                .filterBounds(state_fc)
                .select("label")
                .mode()
                .clip(state_fc)
                .rename(f"{year}_{month_name}_LULC")
            )
            bands.append(lulc)

    image = ee.Image.cat(bands)
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"{state}_X_LULC",
        folder="GEE_Exports",
        fileNamePrefix=f"{state}_X_LULC",
        region=state_fc.geometry(),
        scale=1000,
        crs="EPSG:4326",
        maxPixels=1e13,
    )
    task.start()
    print(f"  started LULC task: {task.id}")


def main() -> None:
    args = parse_args()
    selected_states = selected_states_for_regions(args.regions)
    selected_states.extend(state for state in parse_csv_list(args.extra_full_states) if state not in selected_states)
    selected_set = set(selected_states)
    exclude_full = set(parse_csv_list(args.exclude_full_states))
    exclude_lulc = set(parse_csv_list(args.exclude_lulc_states))

    normals = load_normals(Path(args.normals_csv))
    manifest = load_manifest(Path(args.manifest))

    full_exports: list[tuple[str, str]] = []
    lulc_exports: list[str] = []
    skipped_no_normals: list[str] = []
    not_in_manifest: list[str] = []

    for state in selected_states:
        row = manifest.get(state)
        if row is None:
            not_in_manifest.append(state)
            if state in normals and state not in exclude_full:
                full_exports.append((state, "state_not_in_manifest"))
            elif state not in normals:
                skipped_no_normals.append(state)
            continue

        row["state"] = state
        needs_full, reason = needs_full_export(row)
        if needs_full:
            if state not in normals:
                skipped_no_normals.append(state)
            elif state not in exclude_full:
                full_exports.append((state, reason))
            continue

        if missing_lulc(row) and state not in exclude_lulc:
            lulc_exports.append(state)

    print("Selected regions/states:", ", ".join(selected_states))
    print()
    print("Full exports to submit:")
    for state, reason in full_exports:
        print(f"  - {state}: {reason}")
    print()
    print("LULC-only exports to submit:")
    for state in lulc_exports:
        print(f"  - {state}")
    if skipped_no_normals:
        print()
        print("Skipped because normals are missing:")
        for state in skipped_no_normals:
            print(f"  - {state}")
    if not_in_manifest:
        print()
        print("Not currently in manifest:")
        for state in not_in_manifest:
            print(f"  - {state}")
    print()

    for state, _ in full_exports:
        submit_full_export(state, normals[state], args.project, args.dry_run)
        time.sleep(args.sleep_seconds)

    for state in lulc_exports:
        submit_lulc_export(state, args.project, args.dry_run)
        time.sleep(args.sleep_seconds)

    print("Done.")
    if args.dry_run:
        print("Dry run only; no tasks were submitted.")
    else:
        print("Monitor tasks in the Google Earth Engine Tasks panel.")


if __name__ == "__main__":
    main()
