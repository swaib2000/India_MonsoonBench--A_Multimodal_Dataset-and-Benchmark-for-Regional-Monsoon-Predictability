#!/usr/bin/env python3
"""
Export only missing monthly bands for a state from Google Earth Engine.

This is useful when a state already has the legacy monsoon-only stack:
  - precipitation: 25 bands = 5 years x (Jun, Jul, Aug, Sep, Total)
  - predictors:    20 bands = 5 years x (Jun, Jul, Aug, Sep)

For the Jan-Dec monthly autoregressive benchmark, we only need to add:
  Jan-May and Oct-Dec for each year.

The outputs are intentionally written with a ``<State>_missing_months_*``
prefix so they do not overwrite the existing monsoon files.
"""

from __future__ import annotations

import argparse
import sys

import ee


MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def parse_int_list(text: str) -> list[int]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def export_image(image, state, state_name: str, suffix: str, scale: int = 1000):
    description = f"{state_name}_missing_months_{suffix}"
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder="GEE_Exports",
        fileNamePrefix=description,
        region=state.geometry(),
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e13,
    )
    task.start()
    print(f"Started {suffix}: {task.id}")


def export_missing_months(
    state_name: str,
    project_id: str,
    years: list[int],
    months: list[int],
    include_lulc: bool,
) -> None:
    ee.Initialize(project=project_id)

    states = ee.FeatureCollection("FAO/GAUL/2015/level1")
    state = states.filter(ee.Filter.eq("ADM1_NAME", state_name))

    print(f"Starting missing-month exports for {state_name}")
    print(f"Years: {years}")
    print(f"Months: {months}")
    print("Existing Jun-Sep bands will not be re-exported.")

    precip_bands = []
    lst_bands = []
    ndvi_bands = []
    humidity_bands = []
    wind_bands = []
    soil_bands = []
    lulc_bands = []

    for year in years:
        for month in months:
            month_name = MONTH_NAMES[month]
            start = ee.Date.fromYMD(year, month, 1)
            end = start.advance(1, "month")

            precip = (
                ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterDate(start, end)
                .filterBounds(state)
                .sum()
                .clip(state)
                .rename(f"{year}_{month_name}")
            )
            precip_bands.append(precip)

            terra = (
                ee.ImageCollection("MODIS/061/MOD11A1")
                .filterDate(start, end)
                .filterBounds(state)
                .select("LST_Day_1km")
            )
            aqua = (
                ee.ImageCollection("MODIS/061/MYD11A1")
                .filterDate(start, end)
                .filterBounds(state)
                .select("LST_Day_1km")
            )
            lst = (
                terra.merge(aqua)
                .mean()
                .multiply(0.02)
                .subtract(273.15)
                .clip(state)
                .rename(f"{year}_{month_name}_LST")
            )
            lst_bands.append(lst)

            ndvi = (
                ee.ImageCollection("MODIS/061/MOD13A2")
                .filterDate(start, end)
                .filterBounds(state)
                .select("NDVI")
                .mean()
                .multiply(0.0001)
                .clip(state)
                .rename(f"{year}_{month_name}_NDVI")
            )
            ndvi_bands.append(ndvi)

            era5 = (
                ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                .filterDate(start, end)
                .filterBounds(state)
            )
            humidity = (
                era5.select("surface_pressure", "dewpoint_temperature_2m", "temperature_2m")
                .map(
                    lambda img: ee.Image(100)
                    .multiply(
                        ee.Image(611.21)
                        .multiply(
                            ee.Image.exp(
                                ee.Image(17.502)
                                .multiply(img.select("dewpoint_temperature_2m").subtract(273.15))
                                .divide(img.select("dewpoint_temperature_2m").subtract(32.19))
                            )
                        )
                        .divide(
                            ee.Image(611.21).multiply(
                                ee.Image.exp(
                                    ee.Image(17.502)
                                    .multiply(img.select("temperature_2m").subtract(273.15))
                                    .divide(img.select("temperature_2m").subtract(32.19))
                                )
                            )
                        )
                    )
                    .rename("relative_humidity")
                )
                .mean()
                .clip(state)
                .rename(f"{year}_{month_name}_Humidity")
            )
            humidity_bands.append(humidity)

            wind = (
                era5.select("u_component_of_wind_10m", "v_component_of_wind_10m")
                .map(
                    lambda img: ee.Image.sqrt(
                        img.select("u_component_of_wind_10m")
                        .pow(2)
                        .add(img.select("v_component_of_wind_10m").pow(2))
                    ).rename("wind_speed")
                )
                .mean()
                .clip(state)
                .rename(f"{year}_{month_name}_WindSpeed")
            )
            wind_bands.append(wind)

            soil = (
                ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
                .filterDate(start, end)
                .filterBounds(state)
                .select("volumetric_soil_water_layer_1")
                .mean()
                .clip(state)
                .rename(f"{year}_{month_name}_SoilMoisture")
            )
            soil_bands.append(soil)

            if include_lulc:
                lulc = (
                    ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                    .filterDate(start, end)
                    .filterBounds(state)
                    .select("label")
                    .mode()
                    .clip(state)
                    .rename(f"{year}_{month_name}_LULC")
                )
                lulc_bands.append(lulc)

    export_image(ee.Image.cat(precip_bands), state, state_name, "Y_Precipitation_CHIRPS")
    export_image(ee.Image.cat(lst_bands), state, state_name, "X_LST")
    export_image(ee.Image.cat(ndvi_bands), state, state_name, "X_NDVI")
    export_image(ee.Image.cat(humidity_bands), state, state_name, "X_Relative_Humidity")
    export_image(ee.Image.cat(wind_bands), state, state_name, "X_Wind_Speed")
    export_image(ee.Image.cat(soil_bands), state, state_name, "X_Soil_Moisture")

    if include_lulc:
        export_image(ee.Image.cat(lulc_bands), state, state_name, "X_LULC")

    print("\nSubmitted missing-month export tasks.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export missing monthly GEE bands only.")
    parser.add_argument("--state", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--years", default="2020,2021,2022,2023,2024")
    parser.add_argument(
        "--months",
        default="1,2,3,4,5,10,11,12",
        help="Months to export. Default skips already available Jun-Sep.",
    )
    parser.add_argument(
        "--include-lulc",
        action="store_true",
        help="Also export missing Dynamic World LULC months.",
    )
    args = parser.parse_args()

    try:
        export_missing_months(
            state_name=args.state,
            project_id=args.project,
            years=parse_int_list(args.years),
            months=parse_int_list(args.months),
            include_lulc=args.include_lulc,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
