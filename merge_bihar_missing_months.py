#!/usr/bin/env python3
"""
Merge Bihar's legacy Jun-Sep exports with newly exported missing months.

The old Bihar files contain:
  - predictors: 5 years x 4 monsoon months = 20 bands
  - precipitation: 5 years x (4 monsoon months + monsoon total) = 25 bands

The new GEE exports contain Jan-May and Oct-Dec for each year. This script
creates full Jan-Dec stacks:
  - predictors: 5 years x 12 months = 60 bands
  - precipitation: 5 years x (12 months + annual total) = 65 bands

It keeps a backup of the legacy Bihar rasters before overwriting the canonical
GridData/Bihar/Bihar_*.tif files used by downstream processing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parent
STATE = "Bihar"
STATE_DIR = ROOT / "GridData" / STATE
EXPORT_DIR = ROOT / "GEE_Exports"
BACKUP_DIR = STATE_DIR / "legacy_monsoon_20_25_backup"

YEARS = [2020, 2021, 2022, 2023, 2024]
MONTHS = [
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
MONSOON_MONTHS = {"June", "July", "August", "September"}

NORMALS = "10.7,11.8,8.3,19.5,56.2,167.1,348,281.1,223.9,63.3,7.1,6.5"

PREDICTORS = {
    "LST": ("Bihar_X_LST.tif", "Bihar_missing_months_X_LST.tif", "LST", "LST", "LST"),
    "NDVI": ("Bihar_X_NDVI.tif", "Bihar_missing_months_X_NDVI.tif", "NDVI", "NDVI", "NDVI"),
    "Relative_Humidity": (
        "Bihar_X_Relative_Humidity.tif",
        "Bihar_missing_months_X_Relative_Humidity.tif",
        "RH",
        "Humidity",
        "RH",
    ),
    "Soil_Moisture": (
        "Bihar_X_Soil_Moisture.tif",
        "Bihar_missing_months_X_Soil_Moisture.tif",
        "SM",
        "SoilMoisture",
        "SoilMoisture",
    ),
    "Wind_Speed": (
        "Bihar_X_Wind_Speed.tif",
        "Bihar_missing_months_X_Wind_Speed.tif",
        "Wind",
        "WindSpeed",
        "WindSpeed",
    ),
}


def read_described_bands(path: Path) -> tuple[dict[str, np.ndarray], dict, tuple[int, int]]:
    """Read a multiband raster into a description -> array mapping."""
    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        shape = (src.height, src.width)
        descriptions = src.descriptions
        bands: dict[str, np.ndarray] = {}
        for band_idx, description in enumerate(descriptions, start=1):
            if not description:
                raise ValueError(f"{path} band {band_idx} has no description")
            bands[description] = src.read(band_idx)
    return bands, profile, shape


def backup_existing_files() -> None:
    BACKUP_DIR.mkdir(exist_ok=True)
    suffixes = [
        "_Y_Precipitation_CHIRPS.tif",
        "_Y_Precipitation_GT_geotif.tif",
        "_X_Elevation.tif",
        "_X_Elevation_masked.tif",
        "_X_LST.tif",
        "_X_LST_masked.tif",
        "_X_NDVI.tif",
        "_X_NDVI_masked.tif",
        "_X_Relative_Humidity.tif",
        "_X_Relative_Humidity_masked.tif",
        "_X_Soil_Moisture.tif",
        "_X_Soil_Moisture_masked.tif",
        "_X_Wind_Speed.tif",
        "_X_Wind_Speed_masked.tif",
        "_X_LULC.tif",
        "_X_LULC_masked.tif",
    ]
    for suffix in suffixes:
        src = STATE_DIR / f"{STATE}{suffix}"
        dst = BACKUP_DIR / src.name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def write_stack(path: Path, arrays: list[np.ndarray], descriptions: list[str], profile: dict) -> None:
    if len(arrays) != len(descriptions):
        raise ValueError("arrays/descriptions length mismatch")

    out_profile = profile.copy()
    out_profile.update(count=len(arrays), dtype=arrays[0].dtype)

    with rasterio.open(path, "w", **out_profile) as dst:
        for idx, (array, description) in enumerate(zip(arrays, descriptions), start=1):
            dst.write(array, idx)
            dst.set_band_description(idx, description)
    print(f"Wrote {path} ({len(arrays)} bands)")


def merge_predictor(
    output_name: str,
    legacy_name: str,
    missing_name: str,
    legacy_suffix: str,
    missing_suffix: str,
    output_suffix: str,
) -> None:
    legacy_bands, legacy_profile, legacy_shape = read_described_bands(STATE_DIR / legacy_name)
    missing_bands, _, missing_shape = read_described_bands(EXPORT_DIR / missing_name)
    if legacy_shape != missing_shape:
        raise ValueError(f"Shape mismatch for {output_name}: {legacy_shape} vs {missing_shape}")

    arrays: list[np.ndarray] = []
    descriptions: list[str] = []
    for year in YEARS:
        for month in MONTHS:
            input_suffix = legacy_suffix if month in MONSOON_MONTHS else missing_suffix
            input_description = f"{year}_{month}_{input_suffix}"
            output_description = f"{year}_{month}_{output_suffix}"
            source = legacy_bands if month in MONSOON_MONTHS else missing_bands
            if input_description not in source:
                raise KeyError(f"Missing band {input_description} for {output_name}")
            arrays.append(source[input_description])
            descriptions.append(output_description)

    write_stack(STATE_DIR / output_name, arrays, descriptions, legacy_profile)


def merge_precipitation() -> None:
    legacy_bands, legacy_profile, legacy_shape = read_described_bands(
        STATE_DIR / "Bihar_Y_Precipitation_CHIRPS.tif"
    )
    missing_bands, _, missing_shape = read_described_bands(
        EXPORT_DIR / "Bihar_missing_months_Y_Precipitation_CHIRPS.tif"
    )
    if legacy_shape != missing_shape:
        raise ValueError(f"Precipitation shape mismatch: {legacy_shape} vs {missing_shape}")

    arrays: list[np.ndarray] = []
    descriptions: list[str] = []
    for year in YEARS:
        monthly_arrays: list[np.ndarray] = []
        for month in MONTHS:
            description = f"{year}_{month}"
            source = legacy_bands if month in MONSOON_MONTHS else missing_bands
            if description not in source:
                raise KeyError(f"Missing precipitation band {description}")
            array = source[description].astype(np.float32, copy=False)
            monthly_arrays.append(array)
            arrays.append(array)
            descriptions.append(description)

        stack = np.stack(monthly_arrays, axis=0)
        annual = np.nansum(stack, axis=0).astype(np.float32)
        arrays.append(annual)
        descriptions.append(f"{year}_Total")

    out_profile = legacy_profile.copy()
    out_profile.update(dtype=rasterio.float32, nodata=np.nan)
    write_stack(STATE_DIR / "Bihar_Y_Precipitation_CHIRPS.tif", arrays, descriptions, out_profile)


def collapse_elevation() -> None:
    path = STATE_DIR / "Bihar_X_Elevation.tif"
    bands, profile, _ = read_described_bands(path)
    first_key = next(iter(bands))
    array = bands[first_key]
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=array.dtype)
    write_stack(path, [array], ["Elevation"], out_profile)


def process_bihar() -> None:
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


def audit_outputs() -> None:
    paths = [
        STATE_DIR / "Bihar_Y_Precipitation_CHIRPS.tif",
        STATE_DIR / "Bihar_Y_Precipitation_GT_geotif.tif",
        STATE_DIR / "Bihar_X_Elevation.tif",
        STATE_DIR / "Bihar_X_Elevation_masked.tif",
        STATE_DIR / "Bihar_X_LST.tif",
        STATE_DIR / "Bihar_X_LST_masked.tif",
        STATE_DIR / "Bihar_X_NDVI.tif",
        STATE_DIR / "Bihar_X_NDVI_masked.tif",
        STATE_DIR / "Bihar_X_Relative_Humidity.tif",
        STATE_DIR / "Bihar_X_Relative_Humidity_masked.tif",
        STATE_DIR / "Bihar_X_Soil_Moisture.tif",
        STATE_DIR / "Bihar_X_Soil_Moisture_masked.tif",
        STATE_DIR / "Bihar_X_Wind_Speed.tif",
        STATE_DIR / "Bihar_X_Wind_Speed_masked.tif",
    ]
    print("\nBihar output audit:")
    for path in paths:
        with rasterio.open(path) as src:
            print(f"  {path.relative_to(ROOT)}: bands={src.count}, shape=({src.height}, {src.width})")


def main() -> None:
    if not STATE_DIR.exists():
        raise FileNotFoundError(STATE_DIR)

    backup_existing_files()
    print(f"Backups available in {BACKUP_DIR.relative_to(ROOT)}")

    merge_precipitation()
    for output_key, predictor_info in PREDICTORS.items():
        legacy_name, missing_name, legacy_suffix, missing_suffix, output_suffix = predictor_info
        merge_predictor(
            f"Bihar_X_{output_key}.tif",
            legacy_name,
            missing_name,
            legacy_suffix,
            missing_suffix,
            output_suffix,
        )
    collapse_elevation()
    process_bihar()
    audit_outputs()


if __name__ == "__main__":
    main()
