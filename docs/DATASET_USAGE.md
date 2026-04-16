# Dataset Usage Guide

This guide explains how to inspect and use the released India MonsoonBench archive.

## 1. Download

Download the compressed archive from:

[India MonsoonBench data archive](https://drive.google.com/drive/folders/1w8cE4vUk6ThXHbsKpT9GTpr8nvkQJbF0?usp=drive_link)

The compressed archive is approximately 918.5 MB and expands to about 8.3 GB.

## 2. Extract

If the archive is a `.tar.zst` file:

```bash
tar --use-compress-program=unzstd -xf India_MonsoonBench_DataRelease_2026-04-16.tar.zst
```

Expected layout:

```text
India_MonsoonBench_DataRelease/
  README_DATA.md
  MANIFEST.csv
  metadata/
  results/
  patch_datasets/
    patch_dataset_monthly_ar_northwest_himalayan_full/
    patch_dataset_monthly_ar_central_monsoon_core_full_bihar/
    patch_dataset_monthly_ar_south_peninsular_deccan_full/
    patch_dataset_monthly_ar_east_northeast_humid_orographic_full_assam/
```

## 3. Inspect One Dataset

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

dataset = Path("India_MonsoonBench_DataRelease/patch_datasets/patch_dataset_monthly_ar_northwest_himalayan_full")

config = json.loads((dataset / "config.json").read_text())
samples = pd.read_csv(dataset / "samples.csv")
X = np.load(dataset / "X.npy", mmap_mode="r")
y = np.load(dataset / "y.npy", mmap_mode="r")

print(config)
print(samples.head())
print(X.shape, y.shape)
```

## 4. Split Files

Each regional dataset contains split indices:

| File | Target years |
|---|---|
| `train_idx.npy` | 2021-2022 |
| `val_idx.npy` | 2023 |
| `test_idx.npy` | 2024 |

Example:

```python
train_idx = np.load(dataset / "train_idx.npy")
val_idx = np.load(dataset / "val_idx.npy")
test_idx = np.load(dataset / "test_idx.npy")

print(len(train_idx), len(val_idx), len(test_idx))
```

## 5. Train a Baseline

From the repository root:

```bash
python train_patch_baselines.py \
  --dataset-dir India_MonsoonBench_DataRelease/patch_datasets/patch_dataset_monthly_ar_northwest_himalayan_full \
  --output-dir baseline_runs/tutorial_convlstm_northwest \
  --model convlstm \
  --loss ce \
  --epochs 5 \
  --batch-size 16 \
  --disable-early-stopping
```

Use more epochs for full benchmark reproduction.

## 6. Run the Tutorial Notebook

Open:

```text
notebooks/india_monsoonbench_quickstart.ipynb
```

The notebook loads metadata, finds an extracted regional dataset if available, checks tensor shapes, plots class distributions, and visualizes one predictor channel.
