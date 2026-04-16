#!/usr/bin/env python3
"""
Analyze predictor-predictor correlations across India and by hydroclimatic region.

The script uses an extracted patch dataset (X.npy + samples.csv + config.json),
aggregates each patch into modality-level means, and computes correlation
matrices excluding the precipitation label.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for heatmap visualization") from exc


REGION_GROUPS = {
    "coastal": {
        "Andhra Pradesh",
        "Goa",
        "Gujarat",
        "Karnataka",
        "Kerala",
        "Maharashtra",
        "Odisha",
        "Tamil Nadu",
        "West Bengal",
    },
    "mountain_himalaya": {
        "Himachal Pradesh",
        "Sikkim",
        "Uttarakhand",
    },
    "mountain_northeast": {
        "Arunachal Pradesh",
        "Assam",
        "Manipur",
        "Meghalaya",
        "Mizoram",
        "Nagaland",
        "Tripura",
    },
    "plains_plateau": {
        "Bihar",
        "Chhattisgarh",
        "Haryana",
        "Jharkhand",
        "Madhya Pradesh",
        "Punjab",
        "Rajasthan",
        "Telangana",
        "Uttar Pradesh",
    },
}

MODALITY_LABELS = {
    "elevation": "Elevation",
    "lst": "LST",
    "ndvi": "NDVI",
    "rh": "RH",
    "soil_moisture": "Soil Moisture",
    "wind_speed": "Wind Speed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze predictor correlations by region")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="predictor_correlation_analysis")
    parser.add_argument("--method", default="spearman", choices=["pearson", "spearman", "both"])
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-class-groups",
        action="store_true",
        help="Also write correlations separately for each rainfall class, still excluding the class variable itself.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def read_config(dataset_dir: Path) -> dict:
    import json

    with (dataset_dir / "config.json").open() as f:
        return json.load(f)


def read_samples(dataset_dir: Path) -> list[dict[str, str]]:
    with (dataset_dir / "samples.csv").open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def region_for_state(state: str) -> str:
    for region, states in REGION_GROUPS.items():
        if state in states:
            return region
    return "other"


def modality_from_channel(channel_name: str) -> str:
    if channel_name.startswith("soil_moisture"):
        return "soil_moisture"
    if channel_name.startswith("wind_speed"):
        return "wind_speed"
    if channel_name.startswith("elevation"):
        return "elevation"
    if channel_name.startswith("lst"):
        return "lst"
    if channel_name.startswith("ndvi"):
        return "ndvi"
    if channel_name.startswith("rh"):
        return "rh"
    return channel_name.split("_")[0]


def modality_channel_groups(channel_names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, channel_name in enumerate(channel_names):
        groups[modality_from_channel(channel_name)].append(idx)
    ordered = {}
    for modality in ["elevation", "lst", "ndvi", "rh", "soil_moisture", "wind_speed"]:
        if modality in groups:
            ordered[modality] = groups[modality]
    return ordered


def select_indices(num_samples: int, max_samples: int, seed: int) -> np.ndarray:
    indices = np.arange(num_samples)
    if max_samples > 0 and num_samples > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_samples, replace=False)
        indices.sort()
    return indices


def compute_modality_table(dataset_dir: Path, config: dict, indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
    X = np.load(dataset_dir / "X.npy", mmap_mode="r")
    channel_names = config.get("channel_names", [])
    if not channel_names:
        raise ValueError("config.json is missing channel_names")
    groups = modality_channel_groups(channel_names)
    modalities = list(groups)

    values = np.zeros((len(indices), len(modalities)), dtype=np.float32)
    batch_size = 2048
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = np.asarray(X[batch_indices], dtype=np.float32)
        for modality_idx, modality in enumerate(modalities):
            channel_indices = groups[modality]
            values[start : start + len(batch_indices), modality_idx] = np.nanmean(
                batch[:, channel_indices, :, :],
                axis=(1, 2, 3),
            )
    return values, modalities


def rankdata_2d(values: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(values, dtype=np.float64)
    for col_idx in range(values.shape[1]):
        col = values[:, col_idx]
        order = np.argsort(col, kind="mergesort")
        sorted_col = col[order]
        col_ranks = np.empty(col.shape[0], dtype=np.float64)
        start = 0
        while start < len(sorted_col):
            end = start + 1
            while end < len(sorted_col) and sorted_col[end] == sorted_col[start]:
                end += 1
            avg_rank = 0.5 * (start + end - 1) + 1.0
            col_ranks[order[start:end]] = avg_rank
            start = end
        ranks[:, col_idx] = col_ranks
    return ranks


def correlation_matrix(values: np.ndarray, method: str) -> np.ndarray:
    finite_mask = np.all(np.isfinite(values), axis=1)
    clean = values[finite_mask]
    if clean.shape[0] < 3:
        return np.full((values.shape[1], values.shape[1]), np.nan, dtype=np.float64)
    if method == "spearman":
        clean = rankdata_2d(clean)
    return np.corrcoef(clean, rowvar=False)


def write_matrix_csv(path: Path, matrix: np.ndarray, modalities: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [MODALITY_LABELS.get(modality, modality) for modality in modalities]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["modality", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{value:.6f}" for value in row]])


def plot_heatmap(path: Path, matrix: np.ndarray, modalities: list[str], title: str, dpi: int) -> None:
    labels = [MODALITY_LABELS.get(modality, modality) for modality in modalities]
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=13, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=9)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi)
    plt.close(fig)


def write_modality_summary(path: Path, values: np.ndarray, modalities: list[str], group_values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    groups = ["overall", *sorted(set(group_values))]
    for group in groups:
        if group == "overall":
            mask = np.ones(len(group_values), dtype=bool)
        else:
            mask = np.asarray([value == group for value in group_values])
        if not np.any(mask):
            continue
        subset = values[mask]
        for modality_idx, modality in enumerate(modalities):
            col = subset[:, modality_idx]
            rows.append(
                {
                    "group": group,
                    "modality": MODALITY_LABELS.get(modality, modality),
                    "count": int(np.isfinite(col).sum()),
                    "mean": float(np.nanmean(col)),
                    "std": float(np.nanstd(col)),
                    "median": float(np.nanmedian(col)),
                    "q25": float(np.nanpercentile(col, 25)),
                    "q75": float(np.nanpercentile(col, 75)),
                }
            )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_group(
    output_dir: Path,
    values: np.ndarray,
    modalities: list[str],
    group_name: str,
    mask: np.ndarray,
    methods: list[str],
    dpi: int,
) -> list[dict]:
    rows = []
    subset = values[mask]
    for method in methods:
        matrix = correlation_matrix(subset, method)
        csv_path = output_dir / f"{group_name}_{method}_correlation.csv"
        png_path = output_dir / f"{group_name}_{method}_correlation.png"
        write_matrix_csv(csv_path, matrix, modalities)
        plot_heatmap(
            png_path,
            matrix,
            modalities,
            f"{group_name.replace('_', ' ').title()} Predictor Correlation ({method.title()})",
            dpi,
        )
        rows.append({"group": group_name, "method": method, "count": int(mask.sum()), "csv": str(csv_path), "png": str(png_path)})
    return rows


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_config(dataset_dir)
    samples = read_samples(dataset_dir)
    indices = select_indices(len(samples), args.max_samples, args.seed)
    selected_samples = [samples[int(idx)] for idx in indices]
    region_values = [region_for_state(sample["state"]) for sample in selected_samples]
    class_values = [str(sample["label"]) for sample in selected_samples]

    print(f"Loading modality means for {len(indices)} samples...")
    values, modalities = compute_modality_table(dataset_dir, config, indices)

    methods = ["pearson", "spearman"] if args.method == "both" else [args.method]
    summary_rows = []
    summary_rows.extend(
        analyze_group(
            output_dir,
            values,
            modalities,
            "overall_india",
            np.ones(len(indices), dtype=bool),
            methods,
            args.dpi,
        )
    )

    for region in sorted(set(region_values)):
        mask = np.asarray([value == region for value in region_values])
        summary_rows.extend(analyze_group(output_dir, values, modalities, f"region_{region}", mask, methods, args.dpi))

    if args.include_class_groups:
        for class_id in sorted(set(class_values)):
            mask = np.asarray([value == class_id for value in class_values])
            summary_rows.extend(analyze_group(output_dir, values, modalities, f"class_{class_id}", mask, methods, args.dpi))

    write_modality_summary(output_dir / "modality_summary_by_region.csv", values, modalities, region_values)
    with (output_dir / "outputs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote outputs to {output_dir}")
    print(f"Main heatmap: {output_dir / f'overall_india_{methods[-1]}_correlation.png'}")


if __name__ == "__main__":
    main()
