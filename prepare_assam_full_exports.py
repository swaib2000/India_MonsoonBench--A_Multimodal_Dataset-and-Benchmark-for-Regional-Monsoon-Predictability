#!/usr/bin/env python3
"""
Promote full Assam Jan-Dec GEE exports into GridData/Assam and reprocess.

This repairs the legacy Assam folder that had 20-band predictors and 25-band GT
from the older Jun-Sep workflow. The full exports already exist in GEE_Exports:
  - Assam_Y_Precipitation_CHIRPS.tif: 65 bands
  - Assam_X_* dynamic predictors: 60 bands
  - Assam_X_Elevation.tif: 1 band

The script:
  1. Backs up the legacy Assam rasters.
  2. Copies full exports into GridData/Assam.
  3. Runs process_state_data.py to regenerate GT and masked predictors.
  4. Prints an audit of final band counts.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import rasterio


ROOT = Path(__file__).resolve().parent
STATE = "Assam"
STATE_DIR = ROOT / "GridData" / STATE
EXPORT_DIR = ROOT / "GEE_Exports"
BACKUP_DIR = STATE_DIR / "legacy_monsoon_20_25_backup"
NORMALS = "16.2,32.3,73.8,192.1,313.7,481.4,542.2,405.8,318.1,152.1,28.4,13.2"

FILES = [
    "Assam_Y_Precipitation_CHIRPS.tif",
    "Assam_X_Elevation.tif",
    "Assam_X_LST.tif",
    "Assam_X_NDVI.tif",
    "Assam_X_Relative_Humidity.tif",
    "Assam_X_Soil_Moisture.tif",
    "Assam_X_Wind_Speed.tif",
    "Assam_X_LULC.tif",
]

AUDIT_FILES = [
    "Assam_Y_Precipitation_CHIRPS.tif",
    "Assam_Y_Precipitation_GT_geotif.tif",
    "Assam_X_Elevation.tif",
    "Assam_X_Elevation_masked.tif",
    "Assam_X_LST.tif",
    "Assam_X_LST_masked.tif",
    "Assam_X_NDVI.tif",
    "Assam_X_NDVI_masked.tif",
    "Assam_X_Relative_Humidity.tif",
    "Assam_X_Relative_Humidity_masked.tif",
    "Assam_X_Soil_Moisture.tif",
    "Assam_X_Soil_Moisture_masked.tif",
    "Assam_X_Wind_Speed.tif",
    "Assam_X_Wind_Speed_masked.tif",
]


def backup_existing() -> None:
    BACKUP_DIR.mkdir(exist_ok=True)
    for path in STATE_DIR.glob("Assam_*.tif"):
        dst = BACKUP_DIR / path.name
        if not dst.exists():
            shutil.copy2(path, dst)
    print(f"Backups available in {BACKUP_DIR.relative_to(ROOT)}")


def validate_exports() -> None:
    expected_counts = {
        "Assam_Y_Precipitation_CHIRPS.tif": 65,
        "Assam_X_Elevation.tif": 1,
        "Assam_X_LST.tif": 60,
        "Assam_X_NDVI.tif": 60,
        "Assam_X_Relative_Humidity.tif": 60,
        "Assam_X_Soil_Moisture.tif": 60,
        "Assam_X_Wind_Speed.tif": 60,
        "Assam_X_LULC.tif": 60,
    }
    for name, expected in expected_counts.items():
        path = EXPORT_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        with rasterio.open(path) as src:
            if src.count != expected:
                raise ValueError(f"{path} has {src.count} bands; expected {expected}")


def copy_exports() -> None:
    for name in FILES:
        src = EXPORT_DIR / name
        dst = STATE_DIR / name
        shutil.copy2(src, dst)
        print(f"Copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def process_assam() -> None:
    cmd = [
        sys.executable,
        str(ROOT / "process_state_data.py"),
        "--state",
        STATE,
        "--normals",
        NORMALS,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=STATE_DIR, check=True)


def audit() -> None:
    print("\nAssam output audit:")
    for name in AUDIT_FILES:
        path = STATE_DIR / name
        with rasterio.open(path) as src:
            print(f"  {path.relative_to(ROOT)}: bands={src.count}, shape=({src.height}, {src.width})")


def main() -> None:
    if not STATE_DIR.exists():
        raise FileNotFoundError(STATE_DIR)
    validate_exports()
    backup_existing()
    copy_exports()
    process_assam()
    audit()


if __name__ == "__main__":
    main()
