# Reproducibility Guide

This guide summarizes the end-to-end workflow used for India MonsoonBench.

## 1. Export Earth Observation Rasters

Authenticate Earth Engine and set your GEE project:

```bash
earthengine authenticate
python - <<'PY'
import ee
ee.Initialize(project="YOUR_GEE_PROJECT_ID")
print("Earth Engine initialized")
PY
```

Start exports for a state:

```bash
python export_state_data.py \
  --state "Uttarakhand" \
  --normals "49.6,54.9,54.7,40.5,65.6,184.7,435.8,426.2,204.4,58.4,9.9,21.6" \
  --project YOUR_GEE_PROJECT_ID
```

Downloaded exports should be organized under `GridData/<State>/`.

## 2. Generate Ground Truth and Mask Predictors

```bash
python process_state_data.py --grid-data-dir GridData --normals states_normals.csv
```

This creates rainfall class labels and masked predictor rasters for each state.

## 3. Build Monthly Autoregressive Patch Datasets

```bash
python extract_monthly_autoregressive_patches.py \
  --manifest modeling_manifest.csv \
  --region central_monsoon_core \
  --output-dir patch_dataset_monthly_ar_central_monsoon_core_full_bihar \
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

## 4. Train Baselines

```bash
python train_patch_baselines.py \
  --dataset-dir patch_dataset_monthly_ar_central_monsoon_core_full_bihar \
  --output-dir baseline_runs/convlstm_monthly_ar_central_monsoon_core_ce \
  --model convlstm \
  --loss ce \
  --epochs 30 \
  --batch-size 16 \
  --disable-early-stopping
```

Supported model names include `conv3d`, `convlstm`, and `swin3d`.

## 5. Run Simple Climate Baselines

```bash
python run_monthly_ar_simple_baselines.py \
  --datasets patch_dataset_monthly_ar_central_monsoon_core_full_bihar \
  --output-root baseline_runs/simple_monthly_ar_baselines \
  --run-analysis
```

This evaluates persistence and seasonal climatology baselines.

## 6. Run Diagnostics

```bash
python analyze_patch_predictions.py \
  --dataset-dir patch_dataset_monthly_ar_central_monsoon_core_full_bihar \
  --run-dir baseline_runs/convlstm_monthly_ar_central_monsoon_core_ce

python analyze_extreme_regime_errors.py \
  --dataset-dir patch_dataset_monthly_ar_central_monsoon_core_full_bihar \
  --run-dir baseline_runs/convlstm_monthly_ar_central_monsoon_core_ce

python compute_ordinal_metrics.py \
  --run-dirs baseline_runs/convlstm_monthly_ar_central_monsoon_core_ce \
  --output-csv baseline_runs/convlstm_monthly_ar_central_monsoon_core_ce/ordinal_metrics_summary.csv \
  --write-json

python analyze_transition_specific_performance.py \
  --dataset-dir patch_dataset_monthly_ar_central_monsoon_core_full_bihar \
  --run-dirs baseline_runs/convlstm_monthly_ar_central_monsoon_core_ce \
  --output-dir baseline_runs/convlstm_monthly_ar_central_monsoon_core_ce/transition_analysis
```

## 7. Modality Ablation

```bash
python run_leave_one_modality_ablation_all_regions.py \
  --selection-metric weighted_f1 \
  --epochs 30
```

This runs leave-one-out modality ablations using the best completed model per region.

