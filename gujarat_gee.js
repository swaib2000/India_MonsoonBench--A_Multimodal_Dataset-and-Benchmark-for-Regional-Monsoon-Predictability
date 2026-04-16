// -----------------------------------------------------------------------------
// Google Earth Engine Script for Gujarat Multimodal Weather Dataset
// Generated for: Gujarat
// Precipitation Normals: [50.0, 250.0, 200.0, 50.0] mm (June,July,August,September)
// -----------------------------------------------------------------------------

// 0. Load Indian states boundary and filter for Gujarat
var states = ee.FeatureCollection("FAO/GAUL/2015/level1");
var gujarat = states.filter(ee.Filter.eq('ADM1_NAME', 'Gujarat'));
Map.centerObject(gujarat, 7);
Map.addLayer(gujarat, {color: 'black'}, 'Gujarat');

// 1. Define years and monsoon months
var years = [2020, 2021, 2022, 2023, 2024];
var months = [6, 7, 8, 9];
var monthNames = ['June', 'July', 'August', 'September'];

// =============================================================================
// Y: PRECIPITATION (CHIRPS)
// =============================================================================

// 2. Define CHIRPS-style color palette
var chirpsPalette = ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#2c7fb8', '#225ea8', '#081d58'];

// 3. Separate visualization ranges
var visMonth = {
  min: 0,
  max: 400.0,  // ~400 mm
  palette: chirpsPalette
};

var visTotal = {
  min: 0,
  max: 880.0,  // ~880 mm
  palette: chirpsPalette
};

// 4. Collect rainfall bands
var precipBands = [];

years.forEach(function(year) {
  var monthlyImages = [];

  months.forEach(function(m, idx) {
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var monthly = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterDate(start, end)
                  .filterBounds(gujarat)
                  .sum()
                  .clip(gujarat)
                  .rename(year + '_' + monthNames[idx]);

    monthlyImages.push(monthly);
    precipBands.push(monthly);

    // Visualize monthly rainfall
    Map.addLayer(monthly, visMonth, year + ' ' + monthNames[idx]);
  });

  // Compute total monsoon rainfall
  var total = ee.Image.cat(monthlyImages).reduce(ee.Reducer.sum())
                  .rename(year + '_Total');
  precipBands.push(total);
  Map.addLayer(total, visTotal, year + ' Total');
});

// =============================================================================
// X: LAND SURFACE TEMPERATURE (LST)
// =============================================================================

// 5. Visualization parameters for LST (°C)
var visLST = {
  min: 0,
  max: 50,
  palette: ['navy', 'blue', 'aqua', 'limegreen', 'yellow', 'orange', 'red']
};

// 6. Collect LST bands
var lstBands = [];

years.forEach(function (year) {
  months.forEach(function (m, idx) {
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var terra = ee.ImageCollection('MODIS/061/MOD11A1')
                  .filterDate(start, end)
                  .filterBounds(gujarat)
                  .select('LST_Day_1km');
    var aqua = ee.ImageCollection('MODIS/061/MYD11A1')
                  .filterDate(start, end)
                  .filterBounds(gujarat)
                  .select('LST_Day_1km');

    var monthly = terra.merge(aqua)
                       .mean()
                       .multiply(0.02)
                       .subtract(273.15)
                       .clip(gujarat)
                       .rename(year + '_' + monthNames[idx] + '_LST');

    lstBands.push(monthly);
    Map.addLayer(monthly, visLST, year + ' ' + monthNames[idx] + ' LST');
  });
});

// =============================================================================
// X: NDVI
// =============================================================================

// 7. Visualization parameters for NDVI
var visNDVI = {
  min: -1,
  max: 1,
  palette: ['red', 'yellow', 'green']
};

// 8. Collect NDVI bands
var ndviBands = [];

years.forEach(function (year) {
  months.forEach(function (m, idx) {
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var monthlyNDVI = ee.ImageCollection('MODIS/061/MOD13A2')
                      .filterDate(start, end)
                      .filterBounds(gujarat)
                      .select('NDVI')
                      .mean()
                      .multiply(0.0001)
                      .clip(gujarat)
                      .rename(year + '_' + monthNames[idx] + '_NDVI');

    ndviBands.push(monthlyNDVI);
    Map.addLayer(monthlyNDVI, visNDVI, year + ' ' + monthNames[idx] + ' NDVI');
  });
});

// =============================================================================
// X: RELATIVE HUMIDITY & WIND SPEED (ERA5)
// =============================================================================

// 9. Collect humidity and wind bands
var humidityBands = [];
var windBands = [];

years.forEach(function (year) {
  months.forEach(function (m, idx) {
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var era5Daily = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                    .filterDate(start, end)
                    .filterBounds(gujarat);

    var humidity = era5Daily.select('surface_pressure', 'dewpoint_temperature_2m', 'temperature_2m')
                            .map(function(img) {
                              var sp = img.select('surface_pressure');
                              var td = img.select('dewpoint_temperature_2m');
                              var t = img.select('temperature_2m');
                              var rh = ee.Image(100).multiply(
                                ee.Image(611.21).multiply(ee.Image.exp(ee.Image(17.502).multiply(td.subtract(273.15)).divide(td.subtract(32.19))))
                                .divide(ee.Image(611.21).multiply(ee.Image.exp(ee.Image(17.502).multiply(t.subtract(273.15)).divide(t.subtract(32.19)))))
                              );
                              return rh.rename('relative_humidity');
                            })
                            .mean()
                            .clip(gujarat)
                            .rename(year + '_' + monthNames[idx] + '_Humidity');

    var wind = era5Daily.select('u_component_of_wind_10m', 'v_component_of_wind_10m')
                        .map(function(img) {
                          var u = img.select('u_component_of_wind_10m');
                          var v = img.select('v_component_of_wind_10m');
                          return ee.Image.sqrt(u.pow(2).add(v.pow(2))).rename('wind_speed');
                        })
                        .mean()
                        .clip(gujarat)
                        .rename(year + '_' + monthNames[idx] + '_WindSpeed');

    humidityBands.push(humidity);
    windBands.push(wind);
  });
});

// =============================================================================
// X: SOIL MOISTURE (ERA5)
// =============================================================================

// 10. Collect soil moisture bands
var soilMoistureBands = [];

years.forEach(function (year) {
  months.forEach(function (m, idx) {
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var smColl = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                 .filterDate(start, end)
                 .filterBounds(gujarat)
                 .select('volumetric_soil_water_layer_1')
                 .mean()
                 .clip(gujarat)
                 .rename(year + '_' + monthNames[idx] + '_SoilMoisture');

    soilMoistureBands.push(smColl);
  });
});

// =============================================================================
// X: LAND USE/LAND COVER (Dynamic World)
// =============================================================================

// 11. Collect LULC bands
var lulcBands = [];

years.forEach(function (year) {
  months.forEach(function (m, idx) {
    var start = ee.Date.fromYMD(year, m, 1);
    var end = start.advance(1, 'month');

    var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
             .filterDate(start, end)
             .filterBounds(gujarat)
             .mode()
             .clip(gujarat)
             .rename(year + '_' + monthNames[idx] + '_LULC');

    lulcBands.push(dw);
  });
});

// =============================================================================
// X: ELEVATION (SRTM)
// =============================================================================

// 12. Elevation (static)
var elevation = ee.Image('USGS/SRTMGL1_003')
                .clip(gujarat)
                .rename('Elevation');

// =============================================================================
// COMBINE ALL BANDS
// =============================================================================

// 13. Create final multi-band images
var precipImage = ee.Image.cat(precipBands);  // 25 bands
var lstImage = ee.Image.cat(lstBands);        // 20 bands
var ndviImage = ee.Image.cat(ndviBands);      // 20 bands
var humidityImage = ee.Image.cat(humidityBands); // 20 bands
var windImage = ee.Image.cat(windBands);      // 20 bands
var soilMoistureImage = ee.Image.cat(soilMoistureBands); // 20 bands
var lulcImage = ee.Image.cat(lulcBands);      // 20 bands

// =============================================================================
// EXPORT IMAGES
// =============================================================================

// 14. Export all images
Export.image.toDrive({
  image: precipImage,
  description: 'Gujarat_Y_Precipitation_CHIRPS',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Gujarat_Y_Precipitation_CHIRPS',
  region: gujarat.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: lstImage,
  description: 'Gujarat_X_LST',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Gujarat_X_LST',
  region: gujarat.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: ndviImage,
  description: 'Gujarat_X_NDVI',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Gujarat_X_NDVI',
  region: gujarat.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: humidityImage,
  description: 'Gujarat_X_Relative_Humidity',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Gujarat_X_Relative_Humidity',
  region: gujarat.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: windImage,
  description: 'Gujarat_X_Wind_Speed',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Gujarat_X_Wind_Speed',
  region: gujarat.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: soilMoistureImage,
  description: 'Gujarat_X_Soil_Moisture',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Gujarat_X_Soil_Moisture',
  region: gujarat.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: lulcImage,
  description: 'Gujarat_X_LULC',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Gujarat_X_LULC',
  region: gujarat.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: elevation,
  description: 'Gujarat_X_Elevation',
  folder: 'GEE_Exports',
  fileNamePrefix: 'Gujarat_X_Elevation',
  region: gujarat.geometry(),
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

print('All export tasks initiated for Gujarat');

