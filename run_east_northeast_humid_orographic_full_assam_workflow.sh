#!/usr/bin/env bash
set -euo pipefail

# End-to-end monthly autoregressive rebuttal workflow for the
# East-Northeast Humid Orographic rainfall regime with Assam included.
#
# Formulation:
#   input: 3 previous months, m-3/m-2/m-1
#   target: next monthly rainfall class, m
#   dynamic predictors: precipitation, LST, NDVI, RH, soil moisture, wind speed
#   static predictor: elevation
#   LULC: intentionally excluded
#   split: train=2021-2022, val=2023, test=2024

PYTHON_BIN="${PYTHON_BIN:-python}"

REGION_KEY="east_northeast_humid_orographic"
REGION_NAME="east_northeast_humid_orographic_full_assam"
DATASET_DIR="patch_dataset_monthly_ar_east_northeast_humid_orographic_full_assam"

# Hyperparameters transferred from the best Northwest-Himalayan Optuna trial.
# This avoids another long 24-trial Optuna search while keeping the workflow
# consistent across regional rebuttal experiments.
LR="0.00015591367900525532"
WEIGHT_DECAY="3.6315125458268e-05"
EPOCHS="${EPOCHS:-30}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STATS_SAMPLES="${STATS_SAMPLES:-50000}"

echo "Region: ${REGION_KEY}"
echo "Dataset: ${DATASET_DIR}"
echo "Python: ${PYTHON_BIN}"
echo

if [[ ! -f "${DATASET_DIR}/config.json" || "${FORCE_EXTRACT:-0}" == "1" ]]; then
  echo "Extracting monthly autoregressive patches for ${REGION_KEY} with Assam included..."
  "${PYTHON_BIN}" -u extract_monthly_autoregressive_patches.py \
    --manifest modeling_manifest.csv \
    --region "${REGION_KEY}" \
    --output-dir "${DATASET_DIR}" \
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
    --skip-incomplete-states \
    2>&1 | tee "extract_monthly_ar_${REGION_NAME}_full.log"
else
  echo "Dataset already exists. Set FORCE_EXTRACT=1 to rebuild it."
fi

if [[ ! -f "${DATASET_DIR}/train_idx.npy" || "${FORCE_SPLIT:-0}" == "1" ]]; then
  echo "Creating strict year holdout split..."
  "${PYTHON_BIN}" make_splits.py \
    --dataset-dir "${DATASET_DIR}" \
    --split-mode year_holdout \
    --train-years 2021,2022 \
    --val-years 2023 \
    --test-years 2024
else
  echo "Split already exists. Set FORCE_SPLIT=1 to rebuild it."
fi

run_train_analyze() {
  local model="$1"
  local batch_size="$2"
  local run_dir="baseline_runs/${model}_monthly_ar_${REGION_NAME}_transfer_ce"
  local log_file="train_${model}_monthly_ar_${REGION_NAME}_transfer_ce.log"

  echo
  echo "Training ${model} on ${REGION_NAME}..."
  "${PYTHON_BIN}" train_patch_baselines.py \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${run_dir}" \
    --model "${model}" \
    --loss ce \
    --epochs "${EPOCHS}" \
    --batch-size "${batch_size}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --num-workers "${NUM_WORKERS}" \
    --stats-samples "${STATS_SAMPLES}" \
    --disable-early-stopping \
    2>&1 | tee "${log_file}"

  "${PYTHON_BIN}" analyze_patch_predictions.py \
    --dataset-dir "${DATASET_DIR}" \
    --run-dir "${run_dir}"

  "${PYTHON_BIN}" analyze_extreme_regime_errors.py \
    --dataset-dir "${DATASET_DIR}" \
    --run-dir "${run_dir}"
}

run_train_analyze "conv3d" "32"
run_train_analyze "convlstm" "16"
run_train_analyze "swin3d" "16"

RUN_DIRS=(
  "baseline_runs/conv3d_monthly_ar_${REGION_NAME}_transfer_ce"
  "baseline_runs/convlstm_monthly_ar_${REGION_NAME}_transfer_ce"
  "baseline_runs/swin3d_monthly_ar_${REGION_NAME}_transfer_ce"
)

echo
echo "Computing ordinal-distance metrics..."
"${PYTHON_BIN}" compute_ordinal_metrics.py \
  --run-dirs "${RUN_DIRS[@]}" \
  --output-csv "baseline_runs/${REGION_NAME}_monthly_ar_transfer_ordinal_metrics.csv" \
  --write-json

echo
echo "Computing transition-specific evaluation..."
"${PYTHON_BIN}" analyze_transition_specific_performance.py \
  --dataset-dir "${DATASET_DIR}" \
  --run-dirs "${RUN_DIRS[@]}" \
  --output-dir "baseline_runs/${REGION_NAME}_transition_analysis_transfer_models"

echo
echo "Collecting rebuttal result table..."
"${PYTHON_BIN}" collect_rebuttal_model_results.py \
  --run-dirs "$(IFS=,; echo "${RUN_DIRS[*]}")" \
  --output-csv "baseline_runs/${REGION_NAME}_monthly_ar_results_summary.csv" \
  --output-tex "baseline_runs/${REGION_NAME}_monthly_ar_results_table.tex"

echo
echo "East-Northeast Humid Orographic workflow complete."
echo "Dataset:      ${DATASET_DIR}"
echo "Results CSV:  baseline_runs/${REGION_NAME}_monthly_ar_results_summary.csv"
echo "LaTeX table:  baseline_runs/${REGION_NAME}_monthly_ar_results_table.tex"
echo "Transitions:  baseline_runs/${REGION_NAME}_transition_analysis_transfer_models"
