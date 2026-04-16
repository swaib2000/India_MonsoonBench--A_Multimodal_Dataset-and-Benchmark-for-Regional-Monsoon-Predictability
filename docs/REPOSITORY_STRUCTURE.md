# Repository Structure

This repository keeps executable Python workflows under `scripts/` and keeps research-facing artifacts in documentation folders, result folders, website assets, and notebooks. Commands are written to run from the repository root, for example `python scripts/train_patch_baselines.py ...`.

## Top-Level Research Assets

| Path | Purpose |
|---|---|
| `README.md` | Main research-facing project page |
| `DATA_AVAILABILITY.md` | Data archive link, size table, and excluded artifact policy |
| `REPRODUCIBILITY.md` | End-to-end pipeline commands |
| `CITATION.cff` | Citation metadata |
| `requirements.txt` | Python package requirements |
| `scripts/` | Executable data-processing, training, evaluation, and visualization scripts |
| `results/` | Lightweight CSV benchmark summaries |
| `website/` | Static project website and high-resolution figures |
| `notebooks/` | Tutorial notebook for exploring the released dataset |
| `docs/` | Repository organization and usage guides |

## Script Index

### Earth Engine Export and Data Preparation

| Script | Purpose |
|---|---|
| `scripts/export_state_data.py` | Export a state's multimodal rasters from Google Earth Engine |
| `scripts/export_all_states.py` | Batch state export helper |
| `scripts/export_missing_months_state_data.py` | Export missing monthly bands for partially complete states |
| `scripts/export_missing_states.py` | Export states not present in the current manifest |
| `scripts/export_repair_full_monthly_and_lulc.py` | Repair/export full monthly predictors and optional LULC |
| `scripts/generate_gee_script.py` | Generate Earth Engine helper scripts |
| `scripts/prepare_assam_full_exports.py` | Prepare Assam full-year exports |
| `scripts/merge_bihar_missing_months.py` | Merge Bihar missing monthly bands with existing rasters |
| `scripts/prepare_missing_states_from_exports.py` | Prepare missing states from downloaded exports |

### Raster Processing and Patch Extraction

| Script | Purpose |
|---|---|
| `scripts/batch_process_states.py` | Batch process state rasters |
| `scripts/process_state_data.py` | Mask predictors and generate rainfall anomaly classes |
| `scripts/build_states_normals_from_imd_subdivisions.py` | Build state-level IMD LPA normals |
| `scripts/audit_region_band_counts.py` | Check predictor/target band completeness |
| `scripts/extract_patches.py` | Original static patch extraction |
| `scripts/extract_monthly_autoregressive_patches.py` | Main monthly AR patch extraction |
| `scripts/make_splits.py` | Generate train/validation/test splits |

### Training and Hyperparameter Search

| Script | Purpose |
|---|---|
| `scripts/train_patch_baselines.py` | Train Conv3D, ConvLSTM, Swin3D, and related baselines |
| `scripts/train_cnn_baseline.py` | Train original CNN-style baseline |
| `scripts/tune_patch_baselines_optuna.py` | Optuna hyperparameter tuning |
| `scripts/run_hparam_sweep.py` | Manual sweep runner |
| `scripts/run_region_rebuttal_followups.py` | Region-wise rebuttal workflow |
| `scripts/run_northwest_rebuttal_followups.py` | Northwest-specific follow-up workflow |
| `scripts/run_monthly_ar_simple_baselines.py` | Persistence and seasonal climatology baselines |
| `scripts/run_leave_one_modality_ablation_all_regions.py` | Leave-one-modality-out ablation workflow |
| `scripts/run_season_ablation.py` | Seasonal ablation workflow |

### Evaluation and Diagnostics

| Script | Purpose |
|---|---|
| `scripts/analyze_patch_predictions.py` | Per-state and per-region prediction analysis |
| `scripts/analyze_extreme_regime_errors.py` | Scarcity and Large Excess error analysis |
| `scripts/compute_ordinal_metrics.py` | Ordinal-distance metrics |
| `scripts/analyze_transition_specific_performance.py` | Monsoon-transition evaluation |
| `scripts/analyze_predictor_correlations.py` | Predictor correlation diagnostics |
| `scripts/analyze_learned_climate_regimes.py` | Learned climate-regime analysis |
| `scripts/collect_rebuttal_model_results.py` | Collect model summaries into CSV/LaTeX tables |
| `scripts/summarize_region_modality_ablation.py` | Summarize leave-one-modality ablations |

### Visualization

| Script | Purpose |
|---|---|
| `scripts/visualize_state_modalities.py` | State-level modality plots |
| `scripts/visualize_region_modalities.py` | Region-level modality mosaics |
| `scripts/visualize_all_regions_modalities.py` | Batch region visualization |
| `scripts/visualize_india_modalities.py` | India-wide modality mosaics |
| `scripts/create_readme_modality_collage.py` | Generate README India-wide collage assets |
| `scripts/create_readme_region_modality_collage.py` | Generate README region-specific collage assets |

## Large Files

The following local folders are intentionally ignored by git and distributed through the data archive instead:

| Local path pattern | Reason |
|---|---|
| `GEE_Exports/` | Raw Google Earth Engine GeoTIFF exports |
| `GridData/` | Processed per-state raster stacks |
| `patch_dataset*/` | Large NumPy patch arrays |
| `baseline_runs/` | Checkpoints, logs, and full prediction outputs |
| `*.tar.zst`, `*.zip`, `*.tar.gz` | Packaged data archives |
