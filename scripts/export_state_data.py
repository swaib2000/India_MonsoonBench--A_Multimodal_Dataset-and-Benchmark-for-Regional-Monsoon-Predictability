#!/usr/bin/env python3
"""
Export State Data using Earth Engine Python API

This script exports multimodal weather data for a specified Indian state
directly to Google Drive using the Earth Engine Python API.

Usage:
    python scripts/export_state_data.py --state "Maharashtra" --normals "20,15,20,25,70,200,400,300,200,100,40,20" --project "your-project-id"

Required arguments:
    --state: Name of the Indian state
    --normals: Comma-separated precipitation normals for all 12 months (mm)
    --project: Google Earth Engine project ID

The script will start 8 export tasks to Google Drive (65 precipitation bands, 60 bands each for other modalities).
"""

import argparse
import ee
import sys

def export_state_data(state_name, normals_list, project_id):
    """
    Export all data modalities for the state using EE Python API.
    """
    normals = [float(x.strip()) for x in normals_list.split(',')]
    if len(normals) != 12:
        raise ValueError("Must provide exactly 12 normals (January-December)")

    # Initialize Earth Engine
    ee.Initialize(project=project_id)

    print(f"Starting exports for {state_name}...")

    # Load state boundary
    states = ee.FeatureCollection("FAO/GAUL/2015/level1")
    state = states.filter(ee.Filter.eq('ADM1_NAME', state_name))

    # Define parameters
    years = [2020, 2021, 2022, 2023, 2024]
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    # =============================================================================
    # Y: PRECIPITATION (CHIRPS)
    # =============================================================================

    precip_bands = []
    for year in years:
        monthly_images = []
        for m, month_name in zip(months, month_names):
            start = ee.Date.fromYMD(year, m, 1)
            end = start.advance(1, 'month')

            monthly = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                      .filterDate(start, end) \
                      .filterBounds(state) \
                      .sum() \
                      .clip(state) \
                      .rename(f'{year}_{month_name}')

            monthly_images.append(monthly)
            precip_bands.append(monthly)

        # Annual total
        total = ee.Image.cat(monthly_images).reduce(ee.Reducer.sum()) \
                  .rename(f'{year}_Total')
        precip_bands.append(total)

    precip_image = ee.Image.cat(precip_bands)

    # Export precipitation
    task_precip = ee.batch.Export.image.toDrive(
        image=precip_image,
        description=f'{state_name}_Y_Precipitation_CHIRPS',
        folder='GEE_Exports',
        fileNamePrefix=f'{state_name}_Y_Precipitation_CHIRPS',
        region=state.geometry(),
        scale=1000,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task_precip.start()
    print(f"Started precipitation export: {task_precip.id}")

    # =============================================================================
    # X: LAND SURFACE TEMPERATURE (LST)
    # =============================================================================

    lst_bands = []
    for year in years:
        for m, month_name in zip(months, month_names):
            start = ee.Date.fromYMD(year, m, 1)
            end = start.advance(1, 'month')

            terra = ee.ImageCollection('MODIS/061/MOD11A1') \
                    .filterDate(start, end) \
                    .filterBounds(state) \
                    .select('LST_Day_1km')
            aqua = ee.ImageCollection('MODIS/061/MYD11A1') \
                   .filterDate(start, end) \
                   .filterBounds(state) \
                   .select('LST_Day_1km')

            monthly = terra.merge(aqua) \
                           .mean() \
                           .multiply(0.02) \
                           .subtract(273.15) \
                           .clip(state) \
                           .rename(f'{year}_{month_name}_LST')

            lst_bands.append(monthly)

    lst_image = ee.Image.cat(lst_bands)

    task_lst = ee.batch.Export.image.toDrive(
        image=lst_image,
        description=f'{state_name}_X_LST',
        folder='GEE_Exports',
        fileNamePrefix=f'{state_name}_X_LST',
        region=state.geometry(),
        scale=1000,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task_lst.start()
    print(f"Started LST export: {task_lst.id}")

    # =============================================================================
    # X: NDVI
    # =============================================================================

    ndvi_bands = []
    for year in years:
        for m, month_name in zip(months, month_names):
            start = ee.Date.fromYMD(year, m, 1)
            end = start.advance(1, 'month')

            monthly_ndvi = ee.ImageCollection('MODIS/061/MOD13A2') \
                           .filterDate(start, end) \
                           .filterBounds(state) \
                           .select('NDVI') \
                           .mean() \
                           .multiply(0.0001) \
                           .clip(state) \
                           .rename(f'{year}_{month_name}_NDVI')

            ndvi_bands.append(monthly_ndvi)

    ndvi_image = ee.Image.cat(ndvi_bands)

    task_ndvi = ee.batch.Export.image.toDrive(
        image=ndvi_image,
        description=f'{state_name}_X_NDVI',
        folder='GEE_Exports',
        fileNamePrefix=f'{state_name}_X_NDVI',
        region=state.geometry(),
        scale=1000,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task_ndvi.start()
    print(f"Started NDVI export: {task_ndvi.id}")

    # =============================================================================
    # X: RELATIVE HUMIDITY & WIND SPEED (ERA5)
    # =============================================================================

    humidity_bands = []
    wind_bands = []
    for year in years:
        for m, month_name in zip(months, month_names):
            start = ee.Date.fromYMD(year, m, 1)
            end = start.advance(1, 'month')

            era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
                   .filterDate(start, end) \
                   .filterBounds(state)

            # Humidity calculation
            humidity = era5.select('surface_pressure', 'dewpoint_temperature_2m', 'temperature_2m') \
                          .map(lambda img: ee.Image(100).multiply(
                              ee.Image(611.21).multiply(ee.Image.exp(ee.Image(17.502).multiply(
                                  img.select('dewpoint_temperature_2m').subtract(273.15)
                              ).divide(img.select('dewpoint_temperature_2m').subtract(32.19)))) \
                              .divide(ee.Image(611.21).multiply(ee.Image.exp(ee.Image(17.502).multiply(
                                  img.select('temperature_2m').subtract(273.15)
                              ).divide(img.select('temperature_2m').subtract(32.19)))))
                          ).rename('relative_humidity')) \
                          .mean() \
                          .clip(state) \
                          .rename(f'{year}_{month_name}_Humidity')

            # Wind speed
            wind = era5.select('u_component_of_wind_10m', 'v_component_of_wind_10m') \
                       .map(lambda img: ee.Image.sqrt(
                           img.select('u_component_of_wind_10m').pow(2).add(
                           img.select('v_component_of_wind_10m').pow(2))
                       ).rename('wind_speed')) \
                       .mean() \
                       .clip(state) \
                       .rename(f'{year}_{month_name}_WindSpeed')

            humidity_bands.append(humidity)
            wind_bands.append(wind)

    humidity_image = ee.Image.cat(humidity_bands)
    wind_image = ee.Image.cat(wind_bands)

    task_humidity = ee.batch.Export.image.toDrive(
        image=humidity_image,
        description=f'{state_name}_X_Relative_Humidity',
        folder='GEE_Exports',
        fileNamePrefix=f'{state_name}_X_Relative_Humidity',
        region=state.geometry(),
        scale=1000,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task_humidity.start()
    print(f"Started humidity export: {task_humidity.id}")

    task_wind = ee.batch.Export.image.toDrive(
        image=wind_image,
        description=f'{state_name}_X_Wind_Speed',
        folder='GEE_Exports',
        fileNamePrefix=f'{state_name}_X_Wind_Speed',
        region=state.geometry(),
        scale=1000,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task_wind.start()
    print(f"Started wind export: {task_wind.id}")

    # =============================================================================
    # X: SOIL MOISTURE (ERA5)
    # =============================================================================

    soil_bands = []
    for year in years:
        for m, month_name in zip(months, month_names):
            start = ee.Date.fromYMD(year, m, 1)
            end = start.advance(1, 'month')

            soil = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
                   .filterDate(start, end) \
                   .filterBounds(state) \
                   .select('volumetric_soil_water_layer_1') \
                   .mean() \
                   .clip(state) \
                   .rename(f'{year}_{month_name}_SoilMoisture')

            soil_bands.append(soil)

    soil_image = ee.Image.cat(soil_bands)

    task_soil = ee.batch.Export.image.toDrive(
        image=soil_image,
        description=f'{state_name}_X_Soil_Moisture',
        folder='GEE_Exports',
        fileNamePrefix=f'{state_name}_X_Soil_Moisture',
        region=state.geometry(),
        scale=1000,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task_soil.start()
    print(f"Started soil moisture export: {task_soil.id}")

    # =============================================================================
    # X: LAND USE/LAND COVER (Dynamic World)
    # =============================================================================

    lulc_bands = []
    for year in years:
        for m, month_name in zip(months, month_names):
            start = ee.Date.fromYMD(year, m, 1)
            end = start.advance(1, 'month')

            lulc = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
                   .filterDate(start, end) \
                   .filterBounds(state) \
                   .select('label') \
                   .mode() \
                   .clip(state) \
                   .rename(f'{year}_{month_name}_LULC')

            lulc_bands.append(lulc)

    lulc_image = ee.Image.cat(lulc_bands)

    task_lulc = ee.batch.Export.image.toDrive(
        image=lulc_image,
        description=f'{state_name}_X_LULC',
        folder='GEE_Exports',
        fileNamePrefix=f'{state_name}_X_LULC',
        region=state.geometry(),
        scale=1000,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task_lulc.start()
    print(f"Started LULC export: {task_lulc.id}")

    # =============================================================================
    # X: ELEVATION (SRTM)
    # =============================================================================

    elevation = ee.Image('USGS/SRTMGL1_003').clip(state).rename('Elevation')

    task_elevation = ee.batch.Export.image.toDrive(
        image=elevation,
        description=f'{state_name}_X_Elevation',
        folder='GEE_Exports',
        fileNamePrefix=f'{state_name}_X_Elevation',
        region=state.geometry(),
        scale=1000,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task_elevation.start()
    print(f"Started elevation export: {task_elevation.id}")

    print(f"\nAll 8 export tasks started for {state_name}!")
    print("Check your Google Drive 'GEE_Exports' folder for the downloaded TIFFs.")
    print("Note: Exports may take several minutes to hours depending on data size.")

def main():
    parser = argparse.ArgumentParser(description='Export state multimodal data using EE Python API')
    parser.add_argument('--state', required=True, help='Indian state name')
    parser.add_argument('--normals', required=True, help='Precipitation normals: June,July,August,September (comma-separated, mm)')
    parser.add_argument('--project', required=True, help='Google Earth Engine project ID')

    args = parser.parse_args()

    try:
        export_state_data(args.state, args.normals, args.project)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
