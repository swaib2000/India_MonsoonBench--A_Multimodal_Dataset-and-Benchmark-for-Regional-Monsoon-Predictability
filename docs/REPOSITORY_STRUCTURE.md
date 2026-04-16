# Repository Structure

This repository keeps executable scripts at the project root so the published commands remain simple and backward-compatible. The research artifacts are organized through documentation folders, result folders, website assets, and notebooks.

## Top-Level Research Assets

| Path | Purpose |
|---|---|
| `README.md` | Main research-facing project page |
| `DATA_AVAILABILITY.md` | Data archive link, size table, and excluded artifact policy |
| `REPRODUCIBILITY.md` | End-to-end pipeline commands |
| `CITATION.cff` | Citation metadata |
| `requirements.txt` | Python package requirements |
| `results/` | Lightweight CSV benchmark summaries |
| `website/` | Static project website and high-resolution figures |
| `notebooks/` | Tutorial notebook for exploring the released dataset |
| `docs/` | Repository organization and usage guides |

## Script Index

### Earth Engine Export and Data Preparation

| Script | Purpose |
|---|---|
| `export_state_data.py` | Export a state's multimodal rasters from Google Earth Engine |
| `export_all_states.py` | Batch state export helper |
| `export_missing_months_state_data.py` | Export missing monthly bands for partially complete states |
| `export_missing_states.py` | Export states not present in the current manifest |
| `export_repair_full_monthly_and_lulc.py` | Repair/export full monthly predictors and optional LULC |
| `generate_gee_script.py` | Generate Earth Engine helper scripts |
| `prepare_assam_full_exports.py` | Prepare Assam full-year exports |
| `merge_bihar_missing_months.py` | Merge Bihar missing monthly bands with existing rasters |
| `prepare_missing_states_from_exports.py` | Prepare missing states from downloaded exports |

### Raster Processing and Patch Extraction

| Script | Purpose |
|---|---|
| `batch_process_states.py` | Batch process state rasters |
| `process_state_data.py` | Mask predictors and generate rainfall anomaly classes |
| `build_states_normals_from_imd_subdivisions.py` | Build state-level IMD LPA normals |
| `audit_region_band_counts.py` | Check predictor/target band completeness |
| `extract_patches.py` | Original static patch extraction |
| `extract_monthly_autoregressive_patches.py` | Main monthly AR patch extraction |
| `make_splits.py` | Generate train/validation/test splits |

### Training and Hyperparameter Search

| Script | Purpose |
|---|---|
| `train_patch_baselines.py` | Train Conv3D, ConvLSTM, Swin3D, and related baselines |
| `train_cnn_baseline.py` | Train original CNN-style baseline |
| `tune_patch_baselines_optuna.py` | Optuna hyperparameter tuning |
| `run_hparam_sweep.py` | Manual sweep runner |
| `run_region_rebuttal_followups.py` | Region-wise rebuttal workflow |
| `run_northwest_rebuttal_followups.py` | Northwest-specific follow-up workflow |
| `run_monthly_ar_simple_baselines.py` | Persistence and seasonal climatology baselines |
| `run_leave_one_modality_ablation_all_regions.py` | Leave-one-modality-out ablation workflow |
| `run_season_ablation.py` | Seasonal ablation workflow |

### Evaluation and Diagnostics

| Script | Purpose |
|---|---|
| `analyze_patch_predictions.py` | Per-state and per-region prediction analysis |
| `analyze_extreme_regime_errors.py` | Scarcity and Large Excess error analysis |
| `compute_ordinal_metrics.py` | Ordinal-distance metrics |
| `analyze_transition_specific_performance.py` | Monsoon-transition evaluation |
| `analyze_predictor_correlations.py` | Predictor correlation diagnostics |
| `analyze_learned_climate_regimes.py` | Learned climate-regime analysis |
| `collect_rebuttal_model_results.py` | Collect model summaries into CSV/LaTeX tables |
| `summarize_region_modality_ablation.py` | Summarize leave-one-modality ablations |

### Visualization

| Script | Purpose |
|---|---|
| `visualize_state_modalities.py` | State-level modality plots |
| `visualize_region_modalities.py` | Region-level modality mosaics |
| `visualize_all_regions_modalities.py` | Batch region visualization |
| `visualize_india_modalities.py` | India-wide modality mosaics |
| `create_readme_modality_collage.py` | Generate README India-wide collage assets |
| `create_readme_region_modality_collage.py` | Generate README region-specific collage assets |

## Large Files

The following local folders are intentionally ignored by git and distributed through the data archive instead:

| Local path pattern | Reason |
|---|---|
| `GEE_Exports/` | Raw Google Earth Engine GeoTIFF exports |
| `GridData/` | Processed per-state raster stacks |
| `patch_dataset*/` | Large NumPy patch arrays |
| `baseline_runs/` | Checkpoints, logs, and full prediction outputs |
| `*.tar.zst`, `*.zip`, `*.tar.gz` | Packaged data archives |
