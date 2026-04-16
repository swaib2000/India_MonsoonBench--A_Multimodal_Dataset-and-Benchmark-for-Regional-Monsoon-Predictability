#!/usr/bin/env python3
"""
Build state-level rainfall normals from IMD meteorological sub-division data.

This script supports two IMD-style inputs:
1. A precomputed normals table with columns:
       month, rainfall_type, sub_division, rainfall
2. The free IMD sub-divisional monthly rainfall time series (e.g. 1901-2017)
   with wide columns such as:
       SUBDIVISION, YEAR, JAN, FEB, ..., DEC

For the time-series input, the script computes monthly long-period averages (LPA)
for a user-specified baseline window by taking the arithmetic mean of each month's
rainfall values across the selected years. This is the natural construction implied
by IMD's normal/departure formulation, where departures are computed relative to
normal rainfall for the period.

The source is organized by IMD meteorological sub-division, not always by present-day
state. This script therefore makes the mapping explicit and refuses to silently invent
state values where the sub-division data cannot separate the state cleanly.

Usage with free 1901-2017 time series:
    python build_states_normals_from_imd_subdivisions.py \
        --input sub_divisional_monthly_rainfall_1901_2017.csv \
        --start-year 1971 \
        --end-year 2000 \
        --output states_normals_from_imd.csv

Usage with a precomputed normals table:
    python build_states_normals_from_imd_subdivisions.py \
        --input imd_subdivision_normals.csv \
        --output states_normals_from_imd.csv

Notes:
1. If a state maps to exactly one IMD sub-division, its normals are copied directly.
2. If a state spans multiple clean sub-divisions, the script computes a simple mean by
   default unless explicit weights are added below.
3. If a state cannot be isolated from the IMD sub-division source alone, the script
   marks it unresolved and exits non-zero unless `--allow-ambiguous` is set.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

MONTH_ORDER = [
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

MONTH_TO_COLUMN = {
    "January": "JAN",
    "February": "FEB",
    "March": "MAR",
    "April": "APR",
    "May": "MAY",
    "June": "JUN",
    "July": "JUL",
    "August": "AUG",
    "September": "SEP",
    "October": "OCT",
    "November": "NOV",
    "December": "DEC",
}


# These mappings are explicit so the paper can document the assumptions.
# mode:
# - direct: exact one-to-one mapping
# - mean: simple mean across multiple IMD sub-divisions
# - shared: use the same IMD sub-division normals for each constituent state
# - unresolved: IMD sub-division source cannot isolate this state cleanly
STATE_MAPPINGS = {
    "Andhra Pradesh": {
        "mode": "mean",
        "subdivisions": ["Coastal Andhra Pradesh", "Rayalseema"],
        "note": "State spans two IMD sub-divisions; simple mean is only a fallback.",
    },
    "Arunachal Pradesh": {
        "mode": "direct",
        "subdivisions": ["Arunachal Pradesh"],
        "note": "Exact one-to-one mapping.",
    },
    "Assam": {
        "mode": "shared",
        "subdivisions": ["Assam & Meghalaya"],
        "note": "IMD combines Assam and Meghalaya; the shared sub-division normals are assigned to Assam.",
    },
    "Chhattisgarh": {
        "mode": "direct",
        "subdivisions": ["Chhattisgarh"],
        "note": "Exact one-to-one mapping.",
    },
    "Goa": {
        "mode": "unresolved",
        "subdivisions": ["Konkan & Goa"],
        "note": "IMD combines Goa with Konkan; source cannot isolate Goa.",
    },
    "Gujarat": {
        "mode": "mean",
        "subdivisions": ["Gujarat Region", "Saurashtra & Kutch"],
        "note": "State spans two IMD sub-divisions; simple mean is only a fallback.",
    },
    "Haryana": {
        "mode": "unresolved",
        "subdivisions": ["Haryana Delhi & Chandigarh"],
        "note": "IMD combines Haryana with Chandigarh and Delhi; source cannot isolate Haryana.",
    },
    "Himachal Pradesh": {
        "mode": "direct",
        "subdivisions": ["Himachal Pradesh"],
        "note": "Exact one-to-one mapping.",
    },
    "Jharkhand": {
        "mode": "direct",
        "subdivisions": ["Jharkhand"],
        "note": "Exact one-to-one mapping.",
    },
    "Karnataka": {
        "mode": "mean",
        "subdivisions": ["Coastal Karnataka", "North Interior Karnataka", "South Interior Karnataka"],
        "note": "State spans three IMD sub-divisions; simple mean is used as an unweighted fallback.",
    },
    "Kerala": {
        "mode": "direct",
        "subdivisions": ["Kerala"],
        "note": "Exact one-to-one mapping.",
    },
    "Madhya Pradesh": {
        "mode": "mean",
        "subdivisions": ["East Madhya Pradesh", "West Madhya Pradesh"],
        "note": "State spans two IMD sub-divisions; simple mean is only a fallback.",
    },
    "Maharashtra": {
        "mode": "unresolved",
        "subdivisions": ["Konkan & Goa", "Madhya Maharashtra", "Matathwada", "Vidarbha"],
        "note": "Konkan is merged with Goa in IMD source; state cannot be isolated cleanly.",
    },
    "Manipur": {
        "mode": "shared",
        "subdivisions": ["Naga Mani Mizo Tripura"],
        "note": "IMD combines four states; the shared sub-division normals are assigned to Manipur.",
    },
    "Meghalaya": {
        "mode": "unresolved",
        "subdivisions": ["Assam & Meghalaya"],
        "note": "IMD combines Assam and Meghalaya; source cannot isolate Meghalaya.",
    },
    "Mizoram": {
        "mode": "shared",
        "subdivisions": ["Naga Mani Mizo Tripura"],
        "note": "IMD combines four states; the shared sub-division normals are assigned to Mizoram.",
    },
    "Nagaland": {
        "mode": "shared",
        "subdivisions": ["Naga Mani Mizo Tripura"],
        "note": "IMD combines four states; the shared sub-division normals are assigned to Nagaland.",
    },
    "Odisha": {
        "mode": "direct",
        "subdivisions": ["Orissa"],
        "note": "Exact one-to-one mapping.",
    },
    "Punjab": {
        "mode": "direct",
        "subdivisions": ["Punjab"],
        "note": "Exact one-to-one mapping.",
    },
    "Rajasthan": {
        "mode": "mean",
        "subdivisions": ["East Rajasthan", "West Rajasthan"],
        "note": "State spans two IMD sub-divisions; simple mean is used as an unweighted fallback.",
    },
    "Sikkim": {
        "mode": "unresolved",
        "subdivisions": ["Sub Himalayan West Bengal & Sikkim"],
        "note": "IMD combines Sikkim with Sub-Himalayan West Bengal; source cannot isolate Sikkim.",
    },
    "Tamil Nadu": {
        "mode": "unresolved",
        "subdivisions": ["Tamil Nadu"],
        "note": "IMD combines Tamil Nadu with Puducherry and Karaikal; source cannot isolate Tamil Nadu.",
    },
    "Telangana": {
        "mode": "direct",
        "subdivisions": ["Telangana"],
        "note": "Exact one-to-one mapping.",
    },
    "Tripura": {
        "mode": "shared",
        "subdivisions": ["Naga Mani Mizo Tripura"],
        "note": "IMD combines four states; the shared sub-division normals are assigned to Tripura.",
    },
    "Uttar Pradesh": {
        "mode": "mean",
        "subdivisions": ["East Uttar Pradesh", "West Uttar Pradesh"],
        "note": "State spans two IMD sub-divisions; simple mean is only a fallback.",
    },
    "Uttarakhand": {
        "mode": "direct",
        "subdivisions": ["Uttarakhand"],
        "note": "Exact one-to-one mapping.",
    },
    "West Bengal": {
        "mode": "unresolved",
        "subdivisions": ["Gangetic West Bengal", "Sub Himalayan West Bengal & Sikkim"],
        "note": "Sub-Himalayan West Bengal is merged with Sikkim; source cannot isolate West Bengal cleanly.",
    },
}


def normalize_month(value: str) -> str:
    value = value.strip()
    for month in MONTH_ORDER:
        if value.lower() == month.lower():
            return month
    raise ValueError(f"Unexpected month value: {value}")


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() == "NA":
        return None
    return float(value)


def load_precomputed_subdivision_normals(reader: csv.DictReader) -> dict[str, dict[str, float]]:
    normals: dict[str, dict[str, float]] = defaultdict(dict)

    for row in reader:
        if row["rainfall_type"].strip().lower() != "normal rainfall":
            continue

        month = normalize_month(row["month"])
        subdivision = row["sub_division"].strip()
        rainfall = parse_float(row["rainfall"])
        if rainfall is None:
            continue
        normals[subdivision][month] = rainfall

    return dict(normals)


def load_timeseries_subdivision_normals(
    reader: csv.DictReader, start_year: int, end_year: int
) -> dict[str, dict[str, float]]:
    month_accumulator: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in reader:
        year = int(row["YEAR"].strip())
        if year < start_year or year > end_year:
            continue

        subdivision = row["SUBDIVISION"].strip()
        for month in MONTH_ORDER:
            value = parse_float(row[MONTH_TO_COLUMN[month]])
            if value is not None:
                month_accumulator[subdivision][month].append(value)

    normals: dict[str, dict[str, float]] = {}
    for subdivision, monthly_values in month_accumulator.items():
        normals[subdivision] = {}
        for month in MONTH_ORDER:
            values = monthly_values.get(month, [])
            if not values:
                continue
            normals[subdivision][month] = sum(values) / len(values)

    return normals


def load_subdivision_normals(
    path: str, start_year: int | None = None, end_year: int | None = None
) -> dict[str, dict[str, float]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        precomputed_required = {"month", "rainfall_type", "sub_division", "rainfall"}
        timeseries_required = {"SUBDIVISION", "YEAR", "JAN", "FEB", "MAR", "APR", "MAY",
                               "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"}

        if precomputed_required.issubset(fieldnames):
            return load_precomputed_subdivision_normals(reader)

        if timeseries_required.issubset(fieldnames):
            if start_year is None or end_year is None:
                raise ValueError(
                    "Time-series input detected. Please provide --start-year and --end-year."
                )
            return load_timeseries_subdivision_normals(reader, start_year, end_year)

        raise ValueError(
            "Input CSV format not recognized. Expected either "
            "{month, rainfall_type, sub_division, rainfall} or "
            "{SUBDIVISION, YEAR, JAN..DEC} columns."
        )


def average_months(series_list: list[dict[str, float]]) -> list[float]:
    values = []
    for month in MONTH_ORDER:
        month_values = [series[month] for series in series_list]
        values.append(sum(month_values) / len(month_values))
    return values


def format_normals(values: list[float]) -> str:
    rounded = []
    for value in values:
        if math.isclose(value, round(value), abs_tol=1e-9):
            rounded.append(str(int(round(value))))
        else:
            rounded.append(f"{value:.1f}".rstrip("0").rstrip("."))
    return ",".join(rounded)


def build_state_rows(subdivision_normals: dict[str, dict[str, float]], allow_ambiguous: bool):
    output_rows = []
    unresolved_rows = []

    for state, mapping in STATE_MAPPINGS.items():
        mode = mapping["mode"]
        subdivisions = mapping["subdivisions"]
        note = mapping["note"]

        missing_subdivisions = [name for name in subdivisions if name not in subdivision_normals]
        if missing_subdivisions:
            raise ValueError(
                f"Missing sub-division(s) in input CSV for {state}: {', '.join(missing_subdivisions)}"
            )

        if mode == "unresolved":
            unresolved_rows.append(
                {
                    "state": state,
                    "imd_subdivisions": " | ".join(subdivisions),
                    "reason": note,
                }
            )
            if not allow_ambiguous:
                continue

        series_list = [subdivision_normals[name] for name in subdivisions]
        values = average_months(series_list)
        output_rows.append(
            {
                "state": state,
                "normals": format_normals(values),
            }
        )

    return output_rows, unresolved_rows


def write_csv(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build state rainfall normals from IMD meteorological sub-division normals"
    )
    parser.add_argument("--input", required=True, help="Downloaded IMD/Dataful CSV")
    parser.add_argument(
        "--output",
        default="states_normals_from_imd.csv",
        help="Output CSV path (default: states_normals_from_imd.csv)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="Baseline start year for time-series input (e.g. 1971)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Baseline end year for time-series input (e.g. 2000)",
    )
    parser.add_argument(
        "--allow-ambiguous",
        action="store_true",
        help="Also emit fallback mean values for ambiguous states",
    )
    args = parser.parse_args()

    if (args.start_year is None) ^ (args.end_year is None):
        raise SystemExit("Provide both --start-year and --end-year together.")
    if args.start_year is not None and args.end_year is not None and args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year.")

    subdivision_normals = load_subdivision_normals(
        args.input, start_year=args.start_year, end_year=args.end_year
    )
    output_rows, unresolved_rows = build_state_rows(
        subdivision_normals, allow_ambiguous=args.allow_ambiguous
    )

    write_csv(args.output, output_rows, ["state", "normals"])

    unresolved_path = args.output.replace(".csv", "_unresolved.csv")
    write_csv(unresolved_path, unresolved_rows, ["state", "imd_subdivisions", "reason"])

    if unresolved_rows and not args.allow_ambiguous:
        unresolved_states = ", ".join(row["state"] for row in unresolved_rows)
        raise SystemExit(
            "Stopped before writing ambiguous state values. "
            f"Review {unresolved_path}. Unresolved states: {unresolved_states}"
        )


if __name__ == "__main__":
    main()
