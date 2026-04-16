#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="patch_dataset_monthly_ar_central_monsoon_core_full_bihar"
REGION_NAME="central_monsoon_core_full_bihar"

# Hyperparameters transferred from the best Northwest-Himalayan Optuna trial.
LR="0.00015591367900525532"
WEIGHT_DECAY="3.6315125458268e-05"
EPOCHS="30"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_train_analyze() {
  local model="$1"
  local batch_size="$2"
  local run_dir="baseline_runs/${model}_monthly_ar_${REGION_NAME}_transfer_ce"
  local log_file="train_${model}_monthly_ar_${REGION_NAME}_transfer_ce.log"

  "${PYTHON_BIN}" train_patch_baselines.py \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${run_dir}" \
    --model "${model}" \
    --loss ce \
    --epochs "${EPOCHS}" \
    --batch-size "${batch_size}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
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

"${PYTHON_BIN}" compute_ordinal_metrics.py \
  --run-dirs "${RUN_DIRS[@]}" \
  --output-csv "baseline_runs/${REGION_NAME}_monthly_ar_transfer_ordinal_metrics.csv" \
  --write-json

"${PYTHON_BIN}" analyze_transition_specific_performance.py \
  --dataset-dir "${DATASET_DIR}" \
  --run-dirs "${RUN_DIRS[@]}" \
  --output-dir "baseline_runs/${REGION_NAME}_transition_analysis_transfer_models"

"${PYTHON_BIN}" collect_rebuttal_model_results.py \
  --run-dirs "$(IFS=,; echo "${RUN_DIRS[*]}")" \
  --output-csv "baseline_runs/${REGION_NAME}_monthly_ar_results_summary.csv" \
  --output-tex "baseline_runs/${REGION_NAME}_monthly_ar_results_table.tex"

echo
echo "Central monsoon transfer workflow complete."
echo "Results: baseline_runs/${REGION_NAME}_monthly_ar_results_summary.csv"
echo "Table:   baseline_runs/${REGION_NAME}_monthly_ar_results_table.tex"
echo "Transitions: baseline_runs/${REGION_NAME}_transition_analysis_transfer_models"
