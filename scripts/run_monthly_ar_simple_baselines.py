#!/usr/bin/env python3
"""
Run non-neural baselines for all regional monthly autoregressive datasets.

Baselines:
1. Persistence:
   Predict the previous month's rainfall class for the same spatial patch.
   For target month m, y_hat(s, m) = y(s, m-1).

2. State-month seasonal climatology:
   Predict the most frequent training label for the same state and target month.
   Fallback order: state-month -> state overall -> month overall -> global mode.

The script writes run directories compatible with the existing analysis tools:
  - test_predictions.csv
  - metrics.json
  - per-state/per-region/transition/extreme/ordinal analysis outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = ["Scarcity", "Deficit", "Normal", "Excess", "Large Excess"]

DEFAULT_DATASETS = {
    "northwest_himalayan": "patch_dataset_monthly_ar_northwest_himalayan_full",
    "central_monsoon_core_full_bihar": "patch_dataset_monthly_ar_central_monsoon_core_full_bihar",
    "south_peninsular_deccan": "patch_dataset_monthly_ar_south_peninsular_deccan_full",
    "east_northeast_humid_orographic": "patch_dataset_monthly_ar_east_northeast_humid_orographic_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple monthly AR baselines")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset dirs. Defaults to all four regional monthly AR datasets.",
    )
    parser.add_argument(
        "--output-root",
        default="baseline_runs/simple_monthly_ar_baselines",
        help="Directory for combined outputs and summaries.",
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        default=True,
        help="Run per-state, extreme, ordinal, and transition analysis after prediction.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only write predictions/metrics, skipping downstream analysis.",
    )
    return parser.parse_args()


def safe_divide(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num / den)


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int = 5) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(labels, preds):
        matrix[int(y), int(p)] += 1
    return matrix


def per_class_f1(matrix: np.ndarray) -> list[float]:
    scores = []
    for cls in range(matrix.shape[0]):
        tp = float(matrix[cls, cls])
        fp = float(matrix[:, cls].sum() - matrix[cls, cls])
        fn = float(matrix[cls, :].sum() - matrix[cls, cls])
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        scores.append(safe_divide(2 * precision * recall, precision + recall))
    return scores


def per_class_recall(matrix: np.ndarray) -> list[float]:
    recalls = []
    for cls in range(matrix.shape[0]):
        support = float(matrix[cls, :].sum())
        recalls.append(safe_divide(float(matrix[cls, cls]), support))
    return recalls


def compute_metrics(labels: np.ndarray, preds: np.ndarray, metadata: dict[str, object]) -> dict[str, object]:
    matrix = confusion_matrix(labels, preds)
    f1 = per_class_f1(matrix)
    recalls = per_class_recall(matrix)
    supports = matrix.sum(axis=1)
    errors = np.abs(preds - labels)
    signed = preds - labels
    weighted_f1 = safe_divide(float(np.sum(np.asarray(f1) * supports)), float(supports.sum()))
    metrics = {
        **metadata,
        "test_loss": None,
        "test_acc": float(np.mean(labels == preds)),
        "test_weighted_f1": weighted_f1,
        "test_per_class_f1": f1,
        "num_classes": 5,
        "confusion_matrix": matrix.tolist(),
        "within_1_accuracy": float(np.mean(errors <= 1)),
        "within_2_accuracy": float(np.mean(errors <= 2)),
        "mean_abs_class_error": float(errors.mean()),
        "median_abs_class_error": float(np.median(errors)),
        "severe_error_rate_ge2": float(np.mean(errors >= 2)),
        "opposite_extreme_rate": float(np.mean(((labels == 0) & (preds == 4)) | ((labels == 4) & (preds == 0)))),
        "mean_signed_error": float(signed.mean()),
    }
    for idx, class_name in enumerate(CLASS_NAMES):
        key = class_name.lower().replace(" ", "_")
        metrics[f"{key}_recall"] = recalls[idx]
    return metrics


def mode(values: list[int] | np.ndarray) -> int:
    counts = Counter(int(v) for v in values)
    # Deterministic tie-break: choose the lower ordinal class.
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def previous_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def build_climatology_predictor(samples: pd.DataFrame, train_idx: np.ndarray):
    train = samples.iloc[train_idx]
    state_month_mode = {
        key: mode(group["label"].to_numpy())
        for key, group in train.groupby(["state", "target_month"])
    }
    state_mode = {
        key: mode(group["label"].to_numpy())
        for key, group in train.groupby("state")
    }
    month_mode = {
        key: mode(group["label"].to_numpy())
        for key, group in train.groupby("target_month")
    }
    global_mode = mode(train["label"].to_numpy())

    def predict(row: pd.Series) -> int:
        state = row["state"]
        month = int(row["target_month"])
        if (state, month) in state_month_mode:
            return state_month_mode[(state, month)]
        if state in state_mode:
            return state_mode[state]
        if month in month_mode:
            return month_mode[month]
        return global_mode

    return predict


def build_persistence_predictor(samples: pd.DataFrame, fallback_predictor):
    label_lookup = {}
    for row in samples.itertuples(index=False):
        key = (row.state, int(row.row), int(row.col), int(row.target_year), int(row.target_month))
        label_lookup[key] = int(row.label)

    fallback_count = 0

    def predict(row: pd.Series) -> int:
        nonlocal fallback_count
        prev_year, prev_month = previous_month(int(row["target_year"]), int(row["target_month"]))
        key = (row["state"], int(row["row"]), int(row["col"]), prev_year, prev_month)
        if key in label_lookup:
            return label_lookup[key]
        fallback_count += 1
        return fallback_predictor(row)

    def get_fallback_count() -> int:
        return fallback_count

    return predict, get_fallback_count


def write_predictions(path: Path, sample_ids: np.ndarray, labels: np.ndarray, preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "y_true", "y_pred"])
        writer.writeheader()
        for sample_id, label, pred in zip(sample_ids, labels, preds):
            writer.writerow({"sample_id": int(sample_id), "y_true": int(label), "y_pred": int(pred)})


def run_command(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def analyze(dataset_dir: Path, run_dir: Path) -> None:
    run_command([sys.executable, "scripts/analyze_patch_predictions.py", "--dataset-dir", str(dataset_dir), "--run-dir", str(run_dir)])
    run_command([sys.executable, "scripts/analyze_extreme_regime_errors.py", "--dataset-dir", str(dataset_dir), "--run-dir", str(run_dir)])
    run_command(
        [
            sys.executable,
            "scripts/compute_ordinal_metrics.py",
            "--run-dirs",
            str(run_dir),
            "--output-csv",
            str(run_dir / "ordinal_metrics_summary.csv"),
            "--write-json",
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/analyze_transition_specific_performance.py",
            "--dataset-dir",
            str(dataset_dir),
            "--run-dirs",
            str(run_dir),
            "--output-dir",
            str(run_dir / "transition_analysis"),
        ]
    )


def run_dataset(dataset_dir: Path, output_root: Path, run_analysis: bool) -> list[Path]:
    config_path = dataset_dir / "config.json"
    with config_path.open() as f:
        config = json.load(f)

    region = config.get("region", dataset_dir.name.replace("patch_dataset_monthly_ar_", ""))
    samples = pd.read_csv(dataset_dir / "samples.csv")
    train_idx = np.load(dataset_dir / "train_idx.npy")
    test_idx = np.load(dataset_dir / "test_idx.npy")
    test = samples.iloc[test_idx].copy()
    labels = test["label"].to_numpy(dtype=np.int64)
    sample_ids = test["sample_id"].to_numpy(dtype=np.int64)

    climatology_predictor = build_climatology_predictor(samples, train_idx)
    persistence_predictor, get_fallback_count = build_persistence_predictor(samples, climatology_predictor)

    baseline_specs = [
        ("persistence", persistence_predictor),
        ("seasonal_climatology", climatology_predictor),
    ]
    run_dirs = []
    for baseline_name, predictor in baseline_specs:
        print(f"\nRunning {baseline_name} baseline for {region}...")
        preds = np.asarray([predictor(row) for _, row in test.iterrows()], dtype=np.int64)
        run_dir = Path("baseline_runs") / f"{baseline_name}_monthly_ar_{region}"
        run_dirs.append(run_dir)
        write_predictions(run_dir / "test_predictions.csv", sample_ids, labels, preds)

        metadata = {
            "model": baseline_name,
            "loss": "none",
            "region": region,
            "region_label": config.get("region_label", ""),
            "dataset_dir": str(dataset_dir),
            "num_train": int(train_idx.size),
            "num_val": int(np.load(dataset_dir / "val_idx.npy").size),
            "num_test": int(test_idx.size),
            "in_channels": int(config.get("num_channels", 0)),
            "device": "cpu",
            "baseline_description": (
                "previous-month same-patch rainfall class"
                if baseline_name == "persistence"
                else "state-month training-set majority class"
            ),
        }
        if baseline_name == "persistence":
            metadata["fallback_to_climatology_count"] = int(get_fallback_count())
        metrics = compute_metrics(labels, preds, metadata)
        with (run_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)
        print(
            f"{region} / {baseline_name}: "
            f"acc={metrics['test_acc']:.4f}, wf1={metrics['test_weighted_f1']:.4f}, "
            f"within1={metrics['within_1_accuracy']:.4f}, severe={metrics['severe_error_rate_ge2']:.4f}"
        )
        if run_analysis:
            analyze(dataset_dir, run_dir)
    return run_dirs


def collect_summary(run_dirs: list[Path], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in run_dirs:
        with (run_dir / "metrics.json").open() as f:
            metrics = json.load(f)
        rows.append(
            {
                "run_dir": str(run_dir),
                "region": metrics.get("region", ""),
                "model": metrics.get("model", ""),
                "test_acc": metrics.get("test_acc", ""),
                "test_weighted_f1": metrics.get("test_weighted_f1", ""),
                "within_1_accuracy": metrics.get("within_1_accuracy", ""),
                "mean_abs_class_error": metrics.get("mean_abs_class_error", ""),
                "severe_error_rate_ge2": metrics.get("severe_error_rate_ge2", ""),
                "opposite_extreme_rate": metrics.get("opposite_extreme_rate", ""),
                "scarcity_recall": metrics.get("scarcity_recall", ""),
                "large_excess_recall": metrics.get("large_excess_recall", ""),
                "num_test": metrics.get("num_test", ""),
                "fallback_to_climatology_count": metrics.get("fallback_to_climatology_count", ""),
            }
        )

    summary_path = output_root / "simple_baseline_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {summary_path}")

    tex_path = output_root / "simple_baseline_summary_table.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{tabular}{llrrrrr}\\toprule\n")
        f.write("Region & Baseline & Acc. & wF1 & Within-1 & MAE & Severe \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['region'].replace('_', ' ')} & {row['model'].replace('_', ' ')} & "
                f"{float(row['test_acc']):.3f} & {float(row['test_weighted_f1']):.3f} & "
                f"{float(row['within_1_accuracy']):.3f} & {float(row['mean_abs_class_error']):.3f} & "
                f"{float(row['severe_error_rate_ge2']):.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Wrote {tex_path}")


def main() -> None:
    args = parse_args()
    dataset_dirs = [Path(p) for p in args.datasets] if args.datasets else [Path(p) for p in DEFAULT_DATASETS.values()]
    run_analysis = args.run_analysis and not args.skip_analysis

    all_run_dirs = []
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            raise FileNotFoundError(dataset_dir)
        all_run_dirs.extend(run_dataset(dataset_dir, Path(args.output_root), run_analysis))

    collect_summary(all_run_dirs, Path(args.output_root))


if __name__ == "__main__":
    main()
