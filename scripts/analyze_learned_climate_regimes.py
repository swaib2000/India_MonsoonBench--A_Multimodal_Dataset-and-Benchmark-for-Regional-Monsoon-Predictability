#!/usr/bin/env python3
"""
Cluster learned model embeddings to analyze latent climate regimes.

The script:
1. Loads a trained baseline run and its test split.
2. Extracts penultimate embeddings using a forward hook.
3. Reduces embeddings to 2D with PCA.
4. Clusters embeddings with k-means.
5. Summarizes clusters by rainfall class, region, state, and error status.
6. Writes CSV summaries and PNG visualizations.

This is intended as a lightweight, dependency-minimal learned-regime analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for visualization") from exc

try:
    import torch
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for embedding extraction") from exc

from train_patch_baselines import NpyPatchDataset, build_model


CLASS_NAMES = ["Scarcity", "Deficit", "Normal", "Excess", "Large Excess"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze learned climate regimes from model embeddings")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="", help="Defaults to <run-dir>/learned_regimes")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-clusters", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def region_for_state(state: str) -> str:
    for region, states in REGION_GROUPS.items():
        if state in states:
            return region
    return "other"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_samples(path: Path) -> dict[int, dict[str, str]]:
    rows = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[int(row["sample_id"])] = row
    return rows


def load_predictions(path: Path) -> dict[int, int]:
    preds = {}
    if not path.exists():
        return preds
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds[int(row["sample_id"])] = int(row["y_pred"])
    return preds


def get_hook_module(model: torch.nn.Module) -> torch.nn.Module:
    head = getattr(model, "head", None)
    if isinstance(head, torch.nn.Sequential) and len(head) >= 1:
        return head[0]
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, torch.nn.Sequential) and len(classifier) >= 1:
        return classifier[0]
    return model


def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    captured: list[torch.Tensor] = []

    def hook(_module, inputs, _output):
        tensor = inputs[0]
        if tensor.ndim > 2:
            tensor = torch.flatten(tensor, start_dim=1)
        captured.append(tensor.detach().cpu())

    handle = get_hook_module(model).register_forward_hook(hook)
    model.eval()
    embeddings = []
    labels = []
    preds = []
    sample_ids = []
    seen = 0
    try:
        with torch.no_grad():
            for x, y, ids in loader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                batch_preds = logits.argmax(dim=1).cpu().numpy()
                batch_embeddings = captured.pop(0).numpy()
                embeddings.append(batch_embeddings)
                labels.append(y.numpy())
                preds.append(batch_preds)
                sample_ids.append(ids.numpy())
                seen += int(y.shape[0])
                if max_samples > 0 and seen >= max_samples:
                    break
    finally:
        handle.remove()

    emb = np.concatenate(embeddings, axis=0)
    y = np.concatenate(labels, axis=0)
    pred = np.concatenate(preds, axis=0)
    ids = np.concatenate(sample_ids, axis=0)
    if max_samples > 0:
        emb = emb[:max_samples]
        y = y[:max_samples]
        pred = pred[:max_samples]
        ids = ids[:max_samples]
    return emb, y, pred, ids


def pca_reduce(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    reduced = centered @ components.T
    explained = (singular_values**2) / max(x.shape[0] - 1, 1)
    explained_ratio = explained[:n_components] / max(explained.sum(), 1e-12)
    return reduced, components, explained_ratio


def kmeans(x: np.ndarray, num_clusters: int, seed: int, max_iter: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if x.shape[0] < num_clusters:
        raise ValueError("num_clusters cannot exceed number of samples")
    centers = x[rng.choice(x.shape[0], size=num_clusters, replace=False)].copy()
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(num_clusters):
            mask = labels == cluster_id
            if np.any(mask):
                centers[cluster_id] = x[mask].mean(axis=0)
            else:
                centers[cluster_id] = x[rng.integers(0, x.shape[0])]
    return labels


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_clusters(rows: list[dict], num_clusters: int) -> list[dict]:
    output = []
    for cluster_id in range(num_clusters):
        cluster_rows = [row for row in rows if row["cluster"] == cluster_id]
        if not cluster_rows:
            continue
        class_counts = Counter(row["class_name"] for row in cluster_rows)
        region_counts = Counter(row["region"] for row in cluster_rows)
        state_counts = Counter(row["state"] for row in cluster_rows)
        correct = sum(1 for row in cluster_rows if row["correct"])
        extreme = sum(1 for row in cluster_rows if row["class_id"] in {0, 4})
        output.append(
            {
                "cluster": cluster_id,
                "count": len(cluster_rows),
                "accuracy": correct / len(cluster_rows),
                "extreme_fraction": extreme / len(cluster_rows),
                "dominant_class": class_counts.most_common(1)[0][0],
                "dominant_class_fraction": class_counts.most_common(1)[0][1] / len(cluster_rows),
                "dominant_region": region_counts.most_common(1)[0][0],
                "dominant_region_fraction": region_counts.most_common(1)[0][1] / len(cluster_rows),
                "dominant_state": state_counts.most_common(1)[0][0],
                "dominant_state_fraction": state_counts.most_common(1)[0][1] / len(cluster_rows),
            }
        )
    return output


def display_value(color_key: str, value) -> str:
    region_labels = {
        "coastal": "Coastal",
        "mountain_himalaya": "Himalayan Mountain",
        "mountain_northeast": "Northeast Mountain",
        "plains_plateau": "Plains / Plateau",
        "other": "Other",
    }
    if color_key == "region":
        return region_labels.get(str(value), str(value).replace("_", " ").title())
    if color_key == "correct":
        return "Correct prediction" if value else "Incorrect prediction"
    if color_key == "cluster":
        return f"Cluster {value}"
    return str(value)


def plot_scatter(
    rows: list[dict],
    color_key: str,
    output_path: Path,
    title: str,
    explained_ratio: np.ndarray,
) -> None:
    values = sorted({row[color_key] for row in rows}, key=lambda value: str(value))
    color_map = {value: idx for idx, value in enumerate(values)}
    colors = [color_map[row[color_key]] for row in rows]
    x = [row["pc1"] for row in rows]
    y = [row["pc2"] for row in rows]

    plt.figure(figsize=(9, 6.5))
    scatter = plt.scatter(x, y, c=colors, s=8, alpha=0.75, cmap="tab20")
    handles = []
    for value, idx in color_map.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=display_value(color_key, value),
                markerfacecolor=scatter.cmap(scatter.norm(idx)),
                markersize=6,
            )
        )
    plt.legend(handles=handles, loc="best", fontsize=8, frameon=True, title=legend_title(color_key))
    plt.title(title)
    plt.xlabel(f"Principal component 1 ({explained_ratio[0] * 100:.1f}% variance)")
    plt.ylabel(f"Principal component 2 ({explained_ratio[1] * 100:.1f}% variance)")
    plt.grid(alpha=0.18, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def legend_title(color_key: str) -> str:
    titles = {
        "cluster": "K-means group",
        "region": "Hydroclimatic region",
        "class_name": "True rainfall class",
        "correct": "Prediction status",
    }
    return titles.get(color_key, color_key)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "learned_regimes"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(dataset_dir / "config.json")
    metrics = load_json(run_dir / "metrics.json")
    samples = load_samples(dataset_dir / "samples.csv")

    x_path = dataset_dir / "X.npy"
    y_path = dataset_dir / "y.npy"
    idx_path = dataset_dir / f"{args.split}_idx.npy"
    mean_path = run_dir / "channel_mean.npy"
    std_path = run_dir / "channel_std.npy"
    mask_path = run_dir / "channel_mask.npy"

    mean = np.load(mean_path) if mean_path.exists() else None
    std = np.load(std_path) if std_path.exists() else None
    channel_mask = np.load(mask_path) if mask_path.exists() else None
    dataset = NpyPatchDataset(str(x_path), str(y_path), str(idx_path), mean=mean, std=std, channel_mask=channel_mask)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = int(np.load(x_path, mmap_mode="r").shape[1])
    num_classes = int(metrics.get("num_classes", int(np.load(y_path, mmap_mode="r").max()) + 1))
    model = build_model(metrics["model"], in_channels, num_classes, config).to(device)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    embeddings, labels, preds, sample_ids = extract_embeddings(model, loader, device, args.max_samples)
    reduced, _components, explained_ratio = pca_reduce(embeddings, n_components=2)
    cluster_labels = kmeans(reduced, args.num_clusters, args.seed)

    rows = []
    for idx, sample_id in enumerate(sample_ids):
        sample = samples[int(sample_id)]
        class_id = int(labels[idx])
        pred_id = int(preds[idx])
        state = sample["state"]
        rows.append(
            {
                "sample_id": int(sample_id),
                "pc1": float(reduced[idx, 0]),
                "pc2": float(reduced[idx, 1]),
                "cluster": int(cluster_labels[idx]),
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
                "pred_id": pred_id,
                "pred_name": CLASS_NAMES[pred_id] if pred_id < len(CLASS_NAMES) else f"class_{pred_id}",
                "correct": pred_id == class_id,
                "state": state,
                "region": region_for_state(state),
                "target_year": sample.get("target_year", ""),
            }
        )

    cluster_summary = summarize_clusters(rows, args.num_clusters)
    write_csv(output_dir / "embedding_points.csv", rows)
    write_csv(output_dir / "cluster_summary.csv", cluster_summary)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "dataset_dir": str(dataset_dir),
                "split": args.split,
                "num_samples": len(rows),
                "num_clusters": args.num_clusters,
                "pca_explained_ratio": explained_ratio.tolist(),
            },
            f,
            indent=2,
        )

    plot_scatter(
        rows,
        "cluster",
        output_dir / "clusters_pca.png",
        "Latent Hydroclimatic Regimes from Swin3D Embeddings",
        explained_ratio,
    )
    plot_scatter(
        rows,
        "region",
        output_dir / "regions_pca.png",
        "Swin3D Embedding Space Colored by Hydroclimatic Region",
        explained_ratio,
    )
    plot_scatter(
        rows,
        "class_name",
        output_dir / "classes_pca.png",
        "Swin3D Embedding Space Colored by Rainfall-Regime Class",
        explained_ratio,
    )
    plot_scatter(
        rows,
        "correct",
        output_dir / "errors_pca.png",
        "Swin3D Embedding Space Colored by Prediction Correctness",
        explained_ratio,
    )

    print(f"Wrote {output_dir / 'embedding_points.csv'}")
    print(f"Wrote {output_dir / 'cluster_summary.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote PCA plots to {output_dir}")
    print("\nCluster summary:")
    for row in cluster_summary:
        print(
            f"cluster={row['cluster']} count={row['count']} acc={row['accuracy']:.3f} "
            f"class={row['dominant_class']}({row['dominant_class_fraction']:.2f}) "
            f"region={row['dominant_region']}({row['dominant_region_fraction']:.2f})"
        )


if __name__ == "__main__":
    main()
