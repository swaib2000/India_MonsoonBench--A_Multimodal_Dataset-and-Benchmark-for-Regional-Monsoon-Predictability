#!/usr/bin/env python3
"""
Organize downloaded GEE exports into GridData/<State>/ and run post-processing.

This script assumes exported TIFFs have already been downloaded/synced into
GEE_Exports/. It copies the raw TIFFs into each state folder, copies an
existing SateMask folder if the state folder does not already have one, then
runs process_state_data.py to create:
  - State_Y_Precipitation_GT_geotif.tif
  - State_X_*_masked.tif

Examples:
    python prepare_missing_states_from_exports.py --states "Odisha,Telangana"

    python prepare_missing_states_from_exports.py \
      --states "Odisha,Telangana,Jammu and Kashmir" \
      --jk-normals "JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC"
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_STATES = ["Odisha", "Telangana"]
STATE_DIR_ALIASES = {
    # The manifest and existing repository use GridData/Himachal while the
    # GEE export file prefix remains "Himachal Pradesh".
    "Himachal Pradesh": "Himachal",
}
EXPORT_SUFFIXES = [
    "Y_Precipitation_CHIRPS",
    "X_Elevation",
    "X_LST",
    "X_NDVI",
    "X_Relative_Humidity",
    "X_Soil_Moisture",
    "X_Wind_Speed",
    "X_LULC",
]


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


def find_template_mask(grid_dir: Path) -> Path:
    for candidate in sorted(grid_dir.glob("*/SateMask")):
        shp = candidate / "gadm41_IND_1.shp"
        if shp.exists():
            return candidate
    raise FileNotFoundError("Could not find an existing GridData/*/SateMask/gadm41_IND_1.shp")


def expected_export_paths(export_dir: Path, state: str) -> list[Path]:
    return [export_dir / f"{state}_{suffix}.tif" for suffix in EXPORT_SUFFIXES]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare missing states from downloaded GEE exports.")
    parser.add_argument("--states", help="Comma-separated state list. Default: Odisha,Telangana")
    parser.add_argument("--exports-dir", default="GEE_Exports", help="Downloaded exports directory")
    parser.add_argument("--grid-dir", default="GridData", help="GridData directory")
    parser.add_argument("--normals-csv", default="states_normals.csv", help="CSV with state,normals")
    parser.add_argument("--jk-normals", help="12 comma-separated monthly normals for Jammu and Kashmir")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying/processing")
    args = parser.parse_args()

    export_dir = Path(args.exports_dir)
    grid_dir = Path(args.grid_dir)
    normals_csv = Path(args.normals_csv)

    if not export_dir.exists():
        raise FileNotFoundError(f"Exports directory not found: {export_dir}")
    if not grid_dir.exists():
        raise FileNotFoundError(f"GridData directory not found: {grid_dir}")
    if not normals_csv.exists():
        raise FileNotFoundError(f"Normals CSV not found: {normals_csv}")

    normals_by_state = load_normals(normals_csv)
    if args.jk_normals:
        normals_by_state["Jammu and Kashmir"] = validate_normals(
            args.jk_normals, "Jammu and Kashmir"
        )

    states = parse_states(args.states)
    template_mask = find_template_mask(grid_dir)

    for state in states:
        if state not in normals_by_state:
            raise ValueError(
                f"No normals found for '{state}'. For Jammu and Kashmir, pass --jk-normals."
            )

        missing = [p for p in expected_export_paths(export_dir, state) if not p.exists()]
        # LULC is not used by the current modeling manifest, so allow it to be absent.
        missing_required = [p for p in missing if not p.name.endswith("_X_LULC.tif")]
        if missing_required:
            print(f"\nMissing required exports for {state}:")
            for path in missing_required:
                print(f"  - {path}")
            raise FileNotFoundError(f"Missing required exports for {state}")

        state_dir = grid_dir / STATE_DIR_ALIASES.get(state, state)
        mask_dir = state_dir / "SateMask"

        print(f"\nPreparing {state}")
        print(f"  state_dir: {state_dir}")

        if not args.dry_run:
            state_dir.mkdir(parents=True, exist_ok=True)
            if not mask_dir.exists():
                shutil.copytree(template_mask, mask_dir)

        for src in expected_export_paths(export_dir, state):
            if not src.exists():
                print(f"  skipping optional missing file: {src.name}")
                continue
            dst = state_dir / src.name
            print(f"  copy {src} -> {dst}")
            if not args.dry_run:
                shutil.copy2(src, dst)

        cmd = [
            sys.executable,
            str(Path.cwd() / "process_state_data.py"),
            "--state",
            state,
            "--normals",
            normals_by_state[state],
        ]
        print("  process:", " ".join(f"'{x}'" if " " in x else x for x in cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=state_dir, check=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
