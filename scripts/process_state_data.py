#!/usr/bin/env python3
"""
Processing Script for Multimodal Weather Analytics Dataset

This script processes the downloaded GEE TIFFs for a state:
1. Creates precipitation GT categories from CHIRPS data
2. Applies state boundary masking to all TIFFs
3. Saves processed TIFFs

Usage:
    python scripts/process_state_data.py --state StateName --normals "Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec"

Required arguments:
    --state: Name of the Indian state
    --normals: Comma-separated precipitation normals for all 12 months (mm)

Assumes TIFF files are in the current directory with names like:
    StateName_Y_Precipitation_CHIRPS.tif
    StateName_X_LST.tif
    etc.
And shapefile: SateMask/gadm41_IND_1.shp
"""

import os
import argparse
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.io import MemoryFile

def categorize_precipitation(band_data, normal_value):
    """Categorize precipitation into 5 classes based on normal."""
    cat = np.full(band_data.shape, -1, dtype=np.int8)
    if normal_value <= 0:
        return cat

    mask_valid = ~np.isnan(band_data)
    vals = band_data[mask_valid]
    r = vals / normal_value

    c = np.full(vals.shape, -1, dtype=np.int8)
    c[r < 0.4] = 0   # Scarcity
    c[(r >= 0.4) & (r < 0.8)] = 1  # Deficit
    c[(r >= 0.8) & (r < 1.2)] = 2  # Normal
    c[(r >= 1.2) & (r < 1.6)] = 3  # Excess
    c[r >= 1.6] = 4  # Large Excess

    cat[mask_valid] = c
    return cat

def process_state_data(state_name, normals=None, mask_only=False):
    """Process all TIFFs for the state."""
    print(f"Processing data for {state_name}")

    # File paths
    precip_tiff = f"{state_name}_Y_Precipitation_CHIRPS.tif"
    shapefile_path = "SateMask/gadm41_IND_1.shp"

    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    # Load state boundary
    gdf = gpd.read_file(shapefile_path)
    state_gdf = gdf[gdf["NAME_1"].str.lower() == state_name.lower()]
    if state_gdf.empty:
        raise ValueError(f"State '{state_name}' not found in shapefile")
    state_geom = state_gdf.geometry.values[0]
    geojson_geom = [state_geom.__geo_interface__]

    # List of all X TIFFs
    x_tiffs = [
        f"{state_name}_X_Elevation.tif",
        f"{state_name}_X_LST.tif",
        f"{state_name}_X_NDVI.tif",
        f"{state_name}_X_Relative_Humidity.tif",
        f"{state_name}_X_Soil_Moisture.tif",
        f"{state_name}_X_Wind_Speed.tif",
        f"{state_name}_X_LULC.tif"
    ]

    reference_tiff = None
    if not mask_only:
        if not os.path.exists(precip_tiff):
            raise FileNotFoundError(f"Precipitation TIFF not found: {precip_tiff}")

        # Process precipitation CHIRPS and create GT
        print("Processing precipitation data...")
        with rasterio.open(precip_tiff) as src:
            precip_bands = src.read().astype(np.float32)
            profile = src.profile.copy()
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs

        # Create GT categories (65 bands: 5 years × 13 bands each: 12 months + 1 total)
        gt_bands = np.full(precip_bands.shape, -1, dtype=np.int8)
        for i in range(65):  # 65 bands total
            band_idx = i % 13   # 0-11: months, 12: total

            if band_idx < 12:  # Monthly bands
                normal_val = normals[band_idx]
                gt_bands[i] = categorize_precipitation(precip_bands[i], normal_val)
            else:  # Total band
                total_normal = sum(normals)
                gt_bands[i] = categorize_precipitation(precip_bands[i], total_normal)

        # Save GT TIFF
        gt_profile = profile.copy()
        gt_profile.update(dtype=rasterio.int8, nodata=-1, count=65)
        gt_tiff = f"{state_name}_Y_Precipitation_GT_geotif.tif"
        with rasterio.open(gt_tiff, "w", **gt_profile) as dst:
            dst.write(gt_bands)
        print(f"Saved: {gt_tiff}")
        reference_tiff = precip_tiff
    else:
        for x_tiff in x_tiffs:
            if os.path.exists(x_tiff):
                reference_tiff = x_tiff
                break
        if reference_tiff is None:
            raise FileNotFoundError("No predictor TIFF found for mask-only processing")

        with rasterio.open(reference_tiff) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs

    # Process each X TIFF: mask with state boundary
    for x_tiff in x_tiffs:
        if not os.path.exists(x_tiff):
            print(f"Warning: {x_tiff} not found, skipping")
            continue

        print(f"Processing {x_tiff}...")
        with rasterio.open(x_tiff) as src:
            bands = src.read()
            profile = src.profile.copy()

            # Create mask for state
            mask_array = np.zeros((height, width), dtype=bool)
            with MemoryFile() as memfile:
                with memfile.open(
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype='uint8',
                    transform=transform,
                    crs=crs
                ) as temp_ds:
                    temp_ds.write(np.ones((height, width), dtype=np.uint8), 1)
                    masked, _ = mask(temp_ds, geojson_geom, crop=False, filled=True, nodata=0)
                    mask_array = masked[0].astype(bool)

            # Apply mask: set outside to NaN
            masked_bands = bands.astype(np.float32)
            for i in range(bands.shape[0]):
                band = masked_bands[i]
                band[~mask_array] = np.nan
                masked_bands[i] = band

            # Save masked TIFF
            masked_profile = profile.copy()
            masked_profile.update(dtype=rasterio.float32, nodata=np.nan)
            masked_tiff = x_tiff.replace('.tif', '_masked.tif')
            with rasterio.open(masked_tiff, "w", **masked_profile) as dst:
                dst.write(masked_bands)
            print(f"Saved: {masked_tiff}")

    print(f"Processing complete for {state_name}")

def main():
    parser = argparse.ArgumentParser(description='Process state multimodal data')
    parser.add_argument('--state', required=True, help='Indian state name')
    parser.add_argument('--normals', help='Precipitation normals: Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec (comma-separated, mm)')
    parser.add_argument('--mask-only', action='store_true', help='Skip GT generation and only create masked predictor TIFFs')

    args = parser.parse_args()

    try:
        if args.mask_only:
            process_state_data(args.state, normals=None, mask_only=True)
        else:
            if not args.normals:
                raise ValueError("--normals is required unless --mask-only is used")
            normals = [float(x.strip()) for x in args.normals.split(',')]
            if len(normals) != 12:
                raise ValueError("Must provide exactly 12 normals")
            process_state_data(args.state, normals)
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()
