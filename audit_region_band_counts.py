#!/usr/bin/env python3
"""
Audit raster band counts by climate region.

Use this before full monthly autoregressive extraction to identify states that
do not have the required full-year stacks:
  - GT precipitation category raster: 65 bands
  - dynamic predictors: 60 bands
  - static predictor: at least 1 band

Example:
    python3 audit_region_band_counts.py \
      --regions south_peninsular_deccan,east_northeast_humid_orographic
"""

from __future__ import annotations

import argparse
import csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit raster band counts by region")
    parser.add_argument("--manifest", default="modeling_manifest.csv")
    parser.add_argument(
        "--regions",
        default="south_peninsular_deccan,east_northeast_humid_orographic",
        help="Comma-separated region names, or 'all'",
    )
    parser.add_argument("--expected-gt-bands", type=int, default=65)
    parser.add_argument("--expected-dynamic-bands", type=int, default=60)
    parser.add_argument("--output", default="region_band_count_audit.csv")
    return parser.parse_args()


def raster_count(path: str) -> int | str:
    if not path:
        return "missing_path"
    p = Path(path)
    if not p.exists():
        return "missing_file"
    with rasterio.open(p) as src:
        return int(src.count)


def infer_lulc_path(state: str, gt_path: str) -> str:
    return str(Path(gt_path).parent / f"{state}_X_LULC_masked.tif")


def main() -> None:
    args = parse_args()
    if args.regions.strip().lower() == "all":
        regions = list(REGION_GROUPS)
    else:
        regions = [r.strip() for r in args.regions.split(",") if r.strip()]

    unknown = [r for r in regions if r not in REGION_GROUPS]
    if unknown:
        raise ValueError(f"Unknown region(s): {', '.join(unknown)}")

    state_to_region = {
        state: region for region in regions for state in REGION_GROUPS[region]
    }

    rows = []
    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = row["state"].strip()
            if state not in state_to_region:
                continue

            counts = {
                "gt_bands": raster_count(row["gt"]),
                "elevation_bands": raster_count(row["elevation"]),
                "lulc_bands": raster_count(infer_lulc_path(state, row["gt"])),
            }
            for col in DYNAMIC_COLUMNS:
                counts[f"{col}_bands"] = raster_count(row[col])

            problems = []
            if counts["gt_bands"] != args.expected_gt_bands:
                problems.append(f"gt={counts['gt_bands']}")
            for col in DYNAMIC_COLUMNS:
                key = f"{col}_bands"
                if counts[key] != args.expected_dynamic_bands:
                    problems.append(f"{col}={counts[key]}")
            if counts["lulc_bands"] == "missing_file":
                problems.append("lulc=missing")

            rows.append(
                {
                    "region": state_to_region[state],
                    "state": state,
                    **counts,
                    "full_monthly_ready": "yes" if not problems[:6] else "no",
                    "lulc_ready": "yes" if counts["lulc_bands"] != "missing_file" else "no",
                    "issues": "; ".join(problems),
                }
            )

    present_states = {row["state"] for row in rows}
    for state, region in sorted(state_to_region.items()):
        if state not in present_states:
            rows.append(
                {
                    "region": region,
                    "state": state,
                    "gt_bands": "not_in_manifest",
                    "elevation_bands": "not_in_manifest",
                    "lulc_bands": "not_in_manifest",
                    "lst_bands": "not_in_manifest",
                    "ndvi_bands": "not_in_manifest",
                    "rh_bands": "not_in_manifest",
                    "soil_moisture_bands": "not_in_manifest",
                    "wind_speed_bands": "not_in_manifest",
                    "full_monthly_ready": "no",
                    "lulc_ready": "no",
                    "issues": "state_not_in_manifest",
                }
            )

    fieldnames = [
        "region",
        "state",
        "gt_bands",
        "elevation_bands",
        "lulc_bands",
        "lst_bands",
        "ndvi_bands",
        "rh_bands",
        "soil_moisture_bands",
        "wind_speed_bands",
        "full_monthly_ready",
        "lulc_ready",
        "issues",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    print()
    for region in regions:
        region_rows = [row for row in rows if row["region"] == region]
        print(region)
        for row in sorted(region_rows, key=lambda r: r["state"]):
            status = "READY" if row["full_monthly_ready"] == "yes" else "INCOMPLETE"
            print(f"  {status:10s} {row['state']}: {row['issues'] or 'ok'}")


if __name__ == "__main__":
    main()

