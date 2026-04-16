#!/usr/bin/env python3
"""
Compute ordinal-distance metrics for rainfall-regime classification runs.

Rainfall labels are ordered:
0 Scarcity < 1 Deficit < 2 Normal < 3 Excess < 4 Large Excess

This script complements exact accuracy / weighted F1 with metrics that account
for how far a wrong prediction is from the true class.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = ["Scarcity", "Deficit", "Normal", "Excess", "Large Excess"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ordinal-distance metrics from test predictions")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more run directories containing test_predictions.csv and optionally metrics.json",
    )
    parser.add_argument(
        "--output-csv",
        default="ordinal_metrics_summary.csv",
        help="Combined CSV summary path",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Also write ordinal_metrics.json inside each run directory",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    preds = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"y_true", "y_pred"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            labels.append(int(row["y_true"]))
            preds.append(int(row["y_pred"]))
    if not labels:
        raise ValueError(f"No predictions found in {path}")
    return np.asarray(labels, dtype=np.int64), np.asarray(preds, dtype=np.int64)


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        matrix[int(label), int(pred)] += 1
    return matrix


def per_class_recall(matrix: np.ndarray) -> list[float]:
    recalls = []
    for class_idx in range(matrix.shape[0]):
        support = float(matrix[class_idx].sum())
        recalls.append(safe_divide(float(matrix[class_idx, class_idx]), support))
    return recalls


def per_class_mean_abs_error(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> list[float]:
    values = []
    errors = np.abs(preds - labels)
    for class_idx in range(num_classes):
        mask = labels == class_idx
        values.append(0.0 if not np.any(mask) else float(errors[mask].mean()))
    return values


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    num_classes = int(max(labels.max(), preds.max())) + 1
    errors = np.abs(preds - labels)
    signed_errors = preds - labels
    matrix = confusion_matrix(labels, preds, num_classes)

    within_1 = errors <= 1
    severe = errors >= 2
    opposite_extreme = ((labels == 0) & (preds == num_classes - 1)) | (
        (labels == num_classes - 1) & (preds == 0)
    )
    under_prediction = signed_errors < 0
    over_prediction = signed_errors > 0

    recalls = per_class_recall(matrix)
    class_mae = per_class_mean_abs_error(labels, preds, num_classes)

    metrics = {
        "num_samples": int(labels.size),
        "exact_accuracy": float(np.mean(errors == 0)),
        "within_1_accuracy": float(np.mean(within_1)),
        "within_2_accuracy": float(np.mean(errors <= 2)),
        "mean_abs_class_error": float(errors.mean()),
        "median_abs_class_error": float(np.median(errors)),
        "severe_error_rate_ge2": float(np.mean(severe)),
        "opposite_extreme_rate": float(np.mean(opposite_extreme)),
        "under_prediction_rate": float(np.mean(under_prediction)),
        "over_prediction_rate": float(np.mean(over_prediction)),
        "mean_signed_error": float(signed_errors.mean()),
        "confusion_matrix": matrix.tolist(),
    }

    for idx in range(num_classes):
        class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
        key = class_name.lower().replace(" ", "_")
        metrics[f"{key}_recall"] = recalls[idx]
        metrics[f"{key}_mae"] = class_mae[idx]

    return metrics


def load_base_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open() as f:
        return json.load(f)


def flatten_summary(run_dir: Path, ordinal_metrics: dict, base_metrics: dict) -> dict:
    return {
        "run": run_dir.name,
        "model": base_metrics.get("model", ""),
        "loss": base_metrics.get("loss", ""),
        "test_acc": base_metrics.get("test_acc", ordinal_metrics["exact_accuracy"]),
        "test_weighted_f1": base_metrics.get("test_weighted_f1", ""),
        "within_1_accuracy": ordinal_metrics["within_1_accuracy"],
        "within_2_accuracy": ordinal_metrics["within_2_accuracy"],
        "mean_abs_class_error": ordinal_metrics["mean_abs_class_error"],
        "median_abs_class_error": ordinal_metrics["median_abs_class_error"],
        "severe_error_rate_ge2": ordinal_metrics["severe_error_rate_ge2"],
        "opposite_extreme_rate": ordinal_metrics["opposite_extreme_rate"],
        "under_prediction_rate": ordinal_metrics["under_prediction_rate"],
        "over_prediction_rate": ordinal_metrics["over_prediction_rate"],
        "mean_signed_error": ordinal_metrics["mean_signed_error"],
        "scarcity_recall": ordinal_metrics.get("scarcity_recall", ""),
        "deficit_recall": ordinal_metrics.get("deficit_recall", ""),
        "normal_recall": ordinal_metrics.get("normal_recall", ""),
        "excess_recall": ordinal_metrics.get("excess_recall", ""),
        "large_excess_recall": ordinal_metrics.get("large_excess_recall", ""),
    }


def main() -> None:
    args = parse_args()
    rows = []
    for run_dir_text in args.run_dirs:
        run_dir = Path(run_dir_text)
        predictions_path = run_dir / "test_predictions.csv"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

        labels, preds = load_predictions(predictions_path)
        ordinal_metrics = compute_metrics(labels, preds)
        base_metrics = load_base_metrics(run_dir)
        rows.append(flatten_summary(run_dir, ordinal_metrics, base_metrics))

        if args.write_json:
            with (run_dir / "ordinal_metrics.json").open("w") as f:
                json.dump(ordinal_metrics, f, indent=2)
            print(f"Wrote {run_dir / 'ordinal_metrics.json'}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path}")

    print("\nSummary:")
    for row in rows:
        print(
            f"{row['run']}: exact={float(row['test_acc']):.4f}, "
            f"within1={row['within_1_accuracy']:.4f}, "
            f"MAE={row['mean_abs_class_error']:.4f}, "
            f"severe_ge2={row['severe_error_rate_ge2']:.4f}"
        )


if __name__ == "__main__":
    main()
