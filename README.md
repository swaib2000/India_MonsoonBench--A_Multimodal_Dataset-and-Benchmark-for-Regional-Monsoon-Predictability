# India MonsoonBench

India MonsoonBench is a spatially aligned multimodal dataset and benchmark for regional Indian rainfall anomaly forecasting. The benchmark is designed as a dataset/resource contribution rather than a global weather foundation model. It provides a reproducible workflow for curating Earth-observation predictors, generating IMD long-period-average-based rainfall anomaly labels, extracting spatial patches, and evaluating temporal forecasting baselines across Indian hydroclimatic regimes.

## Benchmark Task

The main benchmark is monthly autoregressive rainfall anomaly forecasting.

| Component | Design |
|---|---|
| Input | Three previous months of multimodal predictors |
| Target | Next-month rainfall anomaly class |
| Formalization | Given X(s,t-2), X(s,t-1), and X(s,t), predict Y(s,t+1) |
| Example windows | Jan-Feb-Mar -> Apr; Feb-Mar-Apr -> May; Oct-Nov-Dec -> Jan of the next year |
| Patch size | 15 x 15 pixels |
| Dynamic predictors | Precipitation history, LST, NDVI, relative humidity, soil moisture, wind speed |
| Static predictor | Elevation |
| Input channels | 6 dynamic predictors x 3 months + 1 static elevation = 19 channels |
| Target classes | Scarcity, Deficit, Normal, Excess, Large Excess |
| Input years | 2020-2024 |
| Target-year split | Train: 2021-2022, validation: 2023, test: 2024 |

Rainfall labels are derived by comparing CHIRPS rainfall against Indian Meteorological Department long-period-average rainfall normals.

## Regional Groups

| Region | States in processed benchmark |
|---|---|
| Northwest-Himalayan | Himachal Pradesh, Rajasthan, Haryana, Punjab, Uttarakhand |
| Central Indian Monsoon Core | Bihar, Chhattisgarh, Jharkhand, Madhya Pradesh, Maharashtra, Uttar Pradesh |
| South Peninsular-Deccan | Karnataka, Kerala, Andhra Pradesh, Goa, Tamil Nadu |
| East-Northeast Humid Orographic | Assam, Arunachal Pradesh, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura, West Bengal |

The current processed release covers 25 states. Jammu and Kashmir, Odisha, and Telangana are documented as missing from the processed benchmark because complete GEE exports/post-processing were not available at the rebuttal deadline.

## Repository Contents

| Path | Purpose |
|---|---|
| `export_state_data.py`, `export_missing_months_state_data.py` | Google Earth Engine export scripts |
| `process_state_data.py` | Rainfall class generation and raster masking |
| `extract_monthly_autoregressive_patches.py` | Monthly AR patch extraction |
| `make_splits.py` | Year-held-out split generation |
| `train_patch_baselines.py` | CNN, Conv3D, ConvLSTM, Swin3D and loss variants |
| `run_region_rebuttal_followups.py` | Regional model training/evaluation workflow |
| `run_monthly_ar_simple_baselines.py` | Persistence and seasonal climatology baselines |
| `compute_ordinal_metrics.py` | Ordinal-distance metrics |
| `analyze_extreme_regime_errors.py` | Scarcity/Large Excess error analysis |
| `analyze_transition_specific_performance.py` | Monsoon-transition evaluation |
| `website/` | Static project page with tables, figures, and benchmark explanation |

Large rasters, NumPy patch datasets, model checkpoints, and logs are intentionally excluded from git. They should be generated locally or distributed through a separate data archive.

## Quick Start

Create an environment:

```bash
conda create -n monsoonbench python=3.10 -y
conda activate monsoonbench
pip install -r requirements.txt
```

Extract one regional monthly AR dataset:

```bash
python extract_monthly_autoregressive_patches.py \
  --manifest modeling_manifest.csv \
  --region northwest_himalayan \
  --output-dir patch_dataset_monthly_ar_northwest_himalayan_full \
  --dynamic-modalities precipitation,lst,ndvi,rh,soil_moisture,wind_speed \
  --static-predictor elevation \
  --target-years 2021,2022,2023,2024 \
  --target-months 1,2,3,4,5,6,7,8,9,10,11,12 \
  --input-window 3 \
  --horizon 1 \
  --patch-size 15 \
  --stride 16 \
  --min-valid-fraction 0.5 \
  --fill-value 0.0 \
  --skip-incomplete-states
```

Train a temporal baseline:

```bash
python train_patch_baselines.py \
  --dataset-dir patch_dataset_monthly_ar_northwest_himalayan_full \
  --output-dir baseline_runs/convlstm_monthly_ar_northwest_himalayan_ce \
  --model convlstm \
  --loss ce \
  --epochs 30 \
  --batch-size 16 \
  --disable-early-stopping
```

Compute diagnostics:

```bash
python analyze_patch_predictions.py \
  --dataset-dir patch_dataset_monthly_ar_northwest_himalayan_full \
  --run-dir baseline_runs/convlstm_monthly_ar_northwest_himalayan_ce

python compute_ordinal_metrics.py \
  --run-dirs baseline_runs/convlstm_monthly_ar_northwest_himalayan_ce \
  --output-csv baseline_runs/convlstm_monthly_ar_northwest_himalayan_ce/ordinal_metrics_summary.csv \
  --write-json
```

## Project Website

The static project website is in `website/`. To preview locally:

```bash
python -m http.server 8080 --directory website
```

Then open `http://localhost:8080`.

## Data Availability

The repository does not include large raster exports or patch arrays. The code documents the full processing pipeline from GEE export to model evaluation. Dataset archives and permanent release links should be added separately before public release.

