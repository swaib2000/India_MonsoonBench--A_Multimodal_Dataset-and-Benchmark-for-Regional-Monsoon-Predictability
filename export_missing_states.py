#!/usr/bin/env python3
"""
Start Google Earth Engine exports for selected missing states.

This is a thin wrapper around export_state_data.py that reads monthly
rainfall normals from states_normals.csv where available. Jammu & Kashmir
is intentionally not given a default normal because it is not present in
the current normals table.

Examples:
    python export_missing_states.py --project YOUR_GEE_PROJECT_ID

    python export_missing_states.py --project YOUR_GEE_PROJECT_ID \
      --states "Odisha,Telangana,Jammu and Kashmir" \
      --jk-normals "JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC"
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_STATES = ["Odisha", "Telangana"]


def load_normals(path: Path) -> dict[str, str]:
    normals: dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            normals[row["state"].strip()] = row["normals"].strip()
    return normals


def parse_states(raw: str | None) -> list[str]:
    if raw is None:
        return DEFAULT_STATES
    states = [s.strip() for s in raw.split(",") if s.strip()]
    if not states:
        raise ValueError("--states was provided but no state names were parsed")
    return states


def validate_normals(raw: str, state: str) -> str:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if len(values) != 12:
        raise ValueError(f"{state} must have exactly 12 monthly normals, got {len(values)}")
    for value in values:
        float(value)
    return ",".join(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start GEE exports for selected missing states.")
    parser.add_argument("--project", required=True, help="Google Earth Engine project ID")
    parser.add_argument(
        "--states",
        help=(
            "Comma-separated state list. Default: Odisha,Telangana. "
            "Use 'Jammu and Kashmir' only after providing --jk-normals."
        ),
    )
    parser.add_argument(
        "--normals-csv",
        default="states_normals.csv",
        help="CSV with columns state,normals. Default: states_normals.csv",
    )
    parser.add_argument(
        "--jk-normals",
        help="12 comma-separated monthly normals for Jammu and Kashmir, Jan-Dec.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Delay between state export submissions to avoid overwhelming GEE.",
    )
    args = parser.parse_args()

    normals_csv = Path(args.normals_csv)
    if not normals_csv.exists():
        raise FileNotFoundError(f"Normals CSV not found: {normals_csv}")

    normals_by_state = load_normals(normals_csv)
    if args.jk_normals:
        normals_by_state["Jammu and Kashmir"] = validate_normals(
            args.jk_normals, "Jammu and Kashmir"
        )

    states = parse_states(args.states)

    for state in states:
        if state not in normals_by_state:
            raise ValueError(
                f"No normals found for '{state}'. For Jammu and Kashmir, pass "
                "--jk-normals with 12 Jan-Dec values before exporting/processing."
            )

    print("Starting GEE exports for:")
    for state in states:
        print(f"  - {state}")
    print()

    for state in states:
        cmd = [
            sys.executable,
            "export_state_data.py",
            "--state",
            state,
            "--normals",
            normals_by_state[state],
            "--project",
            args.project,
        ]
        print("Running:", " ".join(f"'{x}'" if " " in x else x for x in cmd))
        subprocess.run(cmd, check=True)
        time.sleep(args.sleep_seconds)

    print("\nAll requested export tasks were submitted.")
    print("Monitor tasks in the Google Earth Engine Tasks panel.")
    print("Completed files will appear in Google Drive folder: GEE_Exports")


if __name__ == "__main__":
    main()

