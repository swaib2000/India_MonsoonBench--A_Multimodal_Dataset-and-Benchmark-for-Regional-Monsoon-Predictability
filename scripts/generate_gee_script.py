#!/usr/bin/env python3
"""
GEE Script Generator for Multimodal Weather Analytics Dataset

This script generates Google Earth Engine JavaScript code to collect multimodal
weather data for a specified Indian state.

Usage:
    python scripts/generate_gee_script.py --state "StateName" --normals "10,5,15,20,50,150,200,180,150,100,30,10"

Required arguments:
    --state: Name of the Indian state (must match FAO/GAUL ADM1_NAME)
    --normals: Comma-separated precipitation normals for June,July,August,September (mm)

Output:
    Prints the complete GEE JavaScript code to stdout.
    Redirect to a file: python scripts/generate_gee_script.py ... > state_gee_script.js
"""

import argparse
import sys

def generate_gee_script(state_name, normals_list):
    """
    Generate the complete GEE JavaScript code for data collection.
    """
    normals = [float(x.strip()) for x in normals_list.split(',')]
    if len(normals) != 12:
        raise ValueError("Must provide exactly 12 normals (January-December)")

    script = f'''// -----------------------------------------------------------------------------
// Google Earth Engine Script for {state_name} Multimodal Weather Dataset
// Generated for: {state_name}
// Precipitation Normals: {normals} mm (June,July,August,September)
// -----------------------------------------------------------------------------

// 0. Load Indian states boundary and filter for {state_name}
var states = ee.FeatureCollection("FAO/GAUL/2015/level1");
var {state_name.lower().replace(' ', '_')} = states.filter(ee.Filter.eq('ADM1_NAME', '{state_name}'));
Map.centerObject({state_name.lower().replace(' ', '_')}, 7);
Map.addLayer({state_name.lower().replace(' ', '_')}, {{color: 'black'}}, '{state_name}');

// 1. Define years and monsoon months
var years = [2020, 2021, 2022, 2023, 2024];
var months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
var monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

// =============================================================================
// Y: PRECIPITATION (CHIRPS)
// =============================================================================

// 2. Define CHIRPS-style color palette
var chirpsPalette = ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#2c7fb8', '#225ea8', '#081d58'];

// 3. Separate visualization ranges
var visMonth = {{
  min: 0,
  max: {max(normals) * 1.6:.1f},  // ~{max(normals)*1.6:.0f} mm
  palette: chirpsPalette
}};

var visTotal = {{
  min: 0,
  max: {sum(normals) * 1.6:.1f},  // ~{sum(normals)*1.6:.0f} mm
  palette: chirpsPalette
}};

// 4. Collect rainfall bands
var precipBands = [];

years.forEach(function(year) {{
  var monthlyImages = [];

  months.forEach(function(m, idx) {{
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var monthly = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterDate(start, end)
                  .filterBounds({state_name.lower().replace(' ', '_')})
                  .sum()
                  .clip({state_name.lower().replace(' ', '_')})
                  .rename(year + '_' + monthNames[idx]);

    monthlyImages.push(monthly);
    precipBands.push(monthly);

    // Visualize monthly rainfall
    Map.addLayer(monthly, visMonth, year + ' ' + monthNames[idx]);
  }});

  // Compute total monsoon rainfall
  var total = ee.Image.cat(monthlyImages).reduce(ee.Reducer.sum())
                  .rename(year + '_Total');
  precipBands.push(total);
  Map.addLayer(total, visTotal, year + ' Total');
}});

// =============================================================================
// X: LAND SURFACE TEMPERATURE (LST)
// =============================================================================

// 5. Visualization parameters for LST (°C)
var visLST = {{
  min: 0,
  max: 50,
  palette: ['navy', 'blue', 'aqua', 'limegreen', 'yellow', 'orange', 'red']
}};

// 6. Collect LST bands
var lstBands = [];

years.forEach(function (year) {{
  months.forEach(function (m, idx) {{
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var terra = ee.ImageCollection('MODIS/061/MOD11A1')
                  .filterDate(start, end)
                  .filterBounds({state_name.lower().replace(' ', '_')})
                  .select('LST_Day_1km');
    var aqua = ee.ImageCollection('MODIS/061/MYD11A1')
                  .filterDate(start, end)
                  .filterBounds({state_name.lower().replace(' ', '_')})
                  .select('LST_Day_1km');

    var monthly = terra.merge(aqua)
                       .mean()
                       .multiply(0.02)
                       .subtract(273.15)
                       .clip({state_name.lower().replace(' ', '_')})
                       .rename(year + '_' + monthNames[idx] + '_LST');

    lstBands.push(monthly);
    Map.addLayer(monthly, visLST, year + ' ' + monthNames[idx] + ' LST');
  }});
}});

// =============================================================================
// X: NDVI
// =============================================================================

// 7. Visualization parameters for NDVI
var visNDVI = {{
  min: -1,
  max: 1,
  palette: ['red', 'yellow', 'green']
}};

// 8. Collect NDVI bands
var ndviBands = [];

years.forEach(function (year) {{
  months.forEach(function (m, idx) {{
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var monthlyNDVI = ee.ImageCollection('MODIS/061/MOD13A2')
                      .filterDate(start, end)
                      .filterBounds({state_name.lower().replace(' ', '_')})
                      .select('NDVI')
                      .mean()
                      .multiply(0.0001)
                      .clip({state_name.lower().replace(' ', '_')})
                      .rename(year + '_' + monthNames[idx] + '_NDVI');

    ndviBands.push(monthlyNDVI);
    Map.addLayer(monthlyNDVI, visNDVI, year + ' ' + monthNames[idx] + ' NDVI');
  }});
}});

// =============================================================================
// X: RELATIVE HUMIDITY & WIND SPEED (ERA5)
// =============================================================================

// 9. Collect humidity and wind bands
var humidityBands = [];
var windBands = [];

years.forEach(function (year) {{
  months.forEach(function (m, idx) {{
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var era5Daily = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                    .filterDate(start, end)
                    .filterBounds({state_name.lower().replace(' ', '_')});

    var humidity = era5Daily.select('surface_pressure', 'dewpoint_temperature_2m', 'temperature_2m')
                            .map(function(img) {{
                              var sp = img.select('surface_pressure');
                              var td = img.select('dewpoint_temperature_2m');
                              var t = img.select('temperature_2m');
                              var rh = ee.Image(100).multiply(
                                ee.Image(611.21).multiply(ee.Image.exp(ee.Image(17.502).multiply(td.subtract(273.15)).divide(td.subtract(32.19))))
                                .divide(ee.Image(611.21).multiply(ee.Image.exp(ee.Image(17.502).multiply(t.subtract(273.15)).divide(t.subtract(32.19)))))
                              );
                              return rh.rename('relative_humidity');
                            }})
                            .mean()
                            .clip({state_name.lower().replace(' ', '_')})
                            .rename(year + '_' + monthNames[idx] + '_Humidity');

    var wind = era5Daily.select('u_component_of_wind_10m', 'v_component_of_wind_10m')
                        .map(function(img) {{
                          var u = img.select('u_component_of_wind_10m');
                          var v = img.select('v_component_of_wind_10m');
                          return ee.Image.sqrt(u.pow(2).add(v.pow(2))).rename('wind_speed');
                        }})
                        .mean()
                        .clip({state_name.lower().replace(' ', '_')})
                        .rename(year + '_' + monthNames[idx] + '_WindSpeed');

    humidityBands.push(humidity);
    windBands.push(wind);
  }});
}});

// =============================================================================
// X: SOIL MOISTURE (ERA5)
// =============================================================================

// 10. Collect soil moisture bands
var soilMoistureBands = [];

years.forEach(function (year) {{
  months.forEach(function (m, idx) {{
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var smColl = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                 .filterDate(start, end)
                 .filterBounds({state_name.lower().replace(' ', '_')})
                 .select('volumetric_soil_water_layer_1')
                 .mean()
                 .clip({state_name.lower().replace(' ', '_')})
                 .rename(year + '_' + monthNames[idx] + '_SoilMoisture');

    soilMoistureBands.push(smColl);
  }});
}});

// =============================================================================
// X: LAND USE/LAND COVER (Dynamic World)
// =============================================================================

// 11. Collect LULC bands
var lulcBands = [];

years.forEach(function (year) {{
  months.forEach(function (m, idx) {{
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
             .filterDate(start, end)
             .filterBounds({state_name.lower().replace(' ', '_')})
             .select('label')
             .mode()
             .clip({state_name.lower().replace(' ', '_')})
             .rename(year + '_' + monthNames[idx] + '_LULC');

    lulcBands.push(dw);
  }});
}});

// =============================================================================
// X: ELEVATION (SRTM)
// =============================================================================

// 12. Elevation (static)
var elevation = ee.Image('USGS/SRTMGL1_003')
                .clip({state_name.lower().replace(' ', '_')})
                .rename('Elevation');

// =============================================================================
// COMBINE ALL BANDS
// =============================================================================

// 13. Create final multi-band images
var precipImage = ee.Image.cat(precipBands);  // 65 bands (5 years × 13: 12 months + annual total)
var lstImage = ee.Image.cat(lstBands);        // 60 bands (5 years × 12 months)
var ndviImage = ee.Image.cat(ndviBands);      // 60 bands (5 years × 12 months)
var humidityImage = ee.Image.cat(humidityBands); // 60 bands (5 years × 12 months)
var windImage = ee.Image.cat(windBands);      // 60 bands (5 years × 12 months)
var soilMoistureImage = ee.Image.cat(soilMoistureBands); // 60 bands (5 years × 12 months)
var lulcImage = ee.Image.cat(lulcBands);      // 60 bands (5 years × 12 months)

// =============================================================================
// EXPORT IMAGES
// =============================================================================

// 14. Export all images
Export.image.toDrive({{
  image: precipImage,
  description: '{state_name}_Y_Precipitation_CHIRPS',
  folder: 'GEE_Exports',
  fileNamePrefix: '{state_name}_Y_Precipitation_CHIRPS',
  region: {state_name.lower().replace(' ', '_')}.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
}});

Export.image.toDrive({{
  image: lstImage,
  description: '{state_name}_X_LST',
  folder: 'GEE_Exports',
  fileNamePrefix: '{state_name}_X_LST',
  region: {state_name.lower().replace(' ', '_')}.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
}});

Export.image.toDrive({{
  image: ndviImage,
  description: '{state_name}_X_NDVI',
  folder: 'GEE_Exports',
  fileNamePrefix: '{state_name}_X_NDVI',
  region: {state_name.lower().replace(' ', '_')}.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
}});

Export.image.toDrive({{
  image: humidityImage,
  description: '{state_name}_X_Relative_Humidity',
  folder: 'GEE_Exports',
  fileNamePrefix: '{state_name}_X_Relative_Humidity',
  region: {state_name.lower().replace(' ', '_')}.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
}});

Export.image.toDrive({{
  image: windImage,
  description: '{state_name}_X_Wind_Speed',
  folder: 'GEE_Exports',
  fileNamePrefix: '{state_name}_X_Wind_Speed',
  region: {state_name.lower().replace(' ', '_')}.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
}});

Export.image.toDrive({{
  image: soilMoistureImage,
  description: '{state_name}_X_Soil_Moisture',
  folder: 'GEE_Exports',
  fileNamePrefix: '{state_name}_X_Soil_Moisture',
  region: {state_name.lower().replace(' ', '_')}.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
}});

Export.image.toDrive({{
  image: lulcImage,
  description: '{state_name}_X_LULC',
  folder: 'GEE_Exports',
  fileNamePrefix: '{state_name}_X_LULC',
  region: {state_name.lower().replace(' ', '_')}.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
}});

Export.image.toDrive({{
  image: elevation,
  description: '{state_name}_X_Elevation',
  folder: 'GEE_Exports',
  fileNamePrefix: '{state_name}_X_Elevation',
  region: {state_name.lower().replace(' ', '_')}.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
}});

print('All export tasks initiated for {state_name}');
'''

    return script

def main():
    parser = argparse.ArgumentParser(description='Generate GEE script for state data collection')
    parser.add_argument('--state', required=True, help='Indian state name (e.g., "Maharashtra")')
    parser.add_argument('--normals', required=True, help='Precipitation normals: June,July,August,September (comma-separated, mm)')

    args = parser.parse_args()

    try:
        script = generate_gee_script(args.state, args.normals)
        print(script)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
