# Data Availability

Large data artifacts are intentionally not stored in this git repository.

Excluded local artifacts include:

- Google Earth Engine GeoTIFF exports (`GEE_Exports/`)
- Processed per-state rasters (`GridData/`)
- NumPy patch datasets (`patch_dataset*/`)
- Model checkpoints and training outputs (`baseline_runs/`)
- Review PDFs, manuscript PDFs, and private response documents

The repository contains scripts and metadata needed to reproduce the benchmark from raw exports, including:

- GEE export scripts
- IMD LPA normal construction scripts
- Rainfall anomaly class generation
- Raster masking
- Monthly autoregressive patch extraction
- Year-held-out split generation
- Model training and evaluation scripts
- Static website tables and visual summaries

Before public release, dataset archives should be deposited separately using an archival service such as Zenodo, Figshare, OSF, or an institutional repository. The README and website should then be updated with permanent DOI/download links.

