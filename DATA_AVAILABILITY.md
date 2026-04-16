# Data Availability

Large data artifacts are intentionally not stored in this git repository.

The compressed dataset archive is available here:

[India MonsoonBench data archive](https://drive.google.com/drive/folders/1w8cE4vUk6ThXHbsKpT9GTpr8nvkQJbF0?usp=drive_link)

## Dataset Size

| Artifact | Contents | Approx. size |
|---|---|---:|
| Compressed data archive | Packaged release uploaded externally | 918.5 MB |
| Expanded data archive | Patch datasets, metadata, and result summaries | About 8.3 GB |
| Northwest-Himalayan patch dataset | Monthly AR patch arrays and split files | 1.9 GB |
| Central Indian Monsoon Core patch dataset | Monthly AR patch arrays and split files | 3.9 GB |
| South Peninsular-Deccan patch dataset | Monthly AR patch arrays and split files | 1.7 GB |
| East-Northeast Humid Orographic patch dataset | Monthly AR patch arrays and split files | 949 MB |

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
- Lightweight benchmark result summaries in `results/`

For long-term public release, the archive should also be mirrored to a persistent repository such as Zenodo, Figshare, OSF, or an institutional repository. A checksum is recommended so users can verify that their downloaded file exactly matches the released archive.
