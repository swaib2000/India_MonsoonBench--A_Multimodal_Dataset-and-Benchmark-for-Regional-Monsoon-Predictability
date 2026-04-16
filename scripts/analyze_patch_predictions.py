#!/usr/bin/env python3
"""
Analyze saved test predictions by state and region group.

Inputs:
- dataset-dir/samples.csv
- run-dir/test_predictions.csv
- optional run-dir/metrics.json

Outputs inside --run-dir:
- per_state_metrics.csv
- per_region_metrics.csv
- analysis_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


REGION_GROUPS = {
    "Assam": "mountain_northeast",
    "Arunachal Pradesh": "mountain_northeast",
    "Manipur": "mountain_northeast",
    "Meghalaya": "mountain_northeast",
    "Mizoram": "mountain_northeast",
    "Nagaland": "mountain_northeast",
    "Sikkim": "mountain_northeast",
    "Tripura": "mountain_northeast",
    "Himachal Pradesh": "mountain_himalaya",
    "Uttarakhand": "mountain_himalaya",
    "Goa": "coastal",
    "Kerala": "coastal",
    "Tamil Nadu": "coastal",
    "West Bengal": "coastal",
    "Andhra Pradesh": "plains_plateau",
    "Bihar": "plains_plateau",
    "Chhattisgarh": "plains_plateau",
    "Gujarat": "plains_plateau",
    "Haryana": "plains_plateau",
    "Jharkhand": "plains_plateau",
    "Karnataka": "plains_plateau",
    "Madhya Pradesh": "plains_plateau",
    "Maharashtra": "plains_plateau",
    "Punjab": "plains_plateau",
    "Rajasthan": "plains_plateau",
    "Uttar Pradesh": "plains_plateau",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze test predictions by state and region")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def confusion_matrix(labels: list[int], preds: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        matrix[int(y_true), int(y_pred)] += 1
    return matrix


def per_class_f1_from_confusion(matrix: np.ndarray) -> list[float]:
    scores = []
    for cls_idx in range(matrix.shape[0]):
        tp = float(matrix[cls_idx, cls_idx])
        fp = float(matrix[:, cls_idx].sum() - tp)
        fn = float(matrix[cls_idx, :].sum() - tp)
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return scores


def weighted_f1_from_confusion(matrix: np.ndarray) -> float:
    f1_scores = per_class_f1_from_confusion(matrix)
    supports = matrix.sum(axis=1).astype(np.float64)
    total = supports.sum()
    if total == 0:
        return 0.0
    return float(sum(score * support for score, support in zip(f1_scores, supports)) / total)


def compute_metrics(labels: list[int], preds: list[int], num_classes: int) -> dict[str, object]:
    matrix = confusion_matrix(labels, preds, num_classes)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    return {
        "count": total,
        "acc": correct / total if total else 0.0,
        "weighted_f1": weighted_f1_from_confusion(matrix),
        "per_class_f1": per_class_f1_from_confusion(matrix),
        "confusion_matrix": matrix.tolist(),
    }


def write_metrics_csv(path: str, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "name",
        "group_type",
        "count",
        "acc",
        "weighted_f1",
        "scarcity_f1",
        "deficit_f1",
        "normal_f1",
        "excess_f1",
        "large_excess_f1",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    samples_path = os.path.join(args.dataset_dir, "samples.csv")
    predictions_path = os.path.join(args.run_dir, "test_predictions.csv")
    metrics_path = os.path.join(args.run_dir, "metrics.json")

    if not os.path.exists(samples_path):
        raise FileNotFoundError(samples_path)
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(predictions_path)

    sample_lookup: dict[int, dict[str, str]] = {}
    with open(samples_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_lookup[int(row["sample_id"])] = row

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            run_metrics = json.load(f)
        num_classes = int(run_metrics["num_classes"])
    else:
        num_classes = 5

    state_labels: dict[str, list[int]] = defaultdict(list)
    state_preds: dict[str, list[int]] = defaultdict(list)
    region_labels: dict[str, list[int]] = defaultdict(list)
    region_preds: dict[str, list[int]] = defaultdict(list)

    with open(predictions_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = int(row["sample_id"])
            sample = sample_lookup[sample_id]
            state = sample["state"]
            region = REGION_GROUPS.get(state, "other")
            y_true = int(row["y_true"])
            y_pred = int(row["y_pred"])
            state_labels[state].append(y_true)
            state_preds[state].append(y_pred)
            region_labels[region].append(y_true)
            region_preds[region].append(y_pred)

    state_rows = []
    state_summary = {}
    for state in sorted(state_labels):
        metrics = compute_metrics(state_labels[state], state_preds[state], num_classes)
        state_summary[state] = metrics
        per_class = metrics["per_class_f1"]
        state_rows.append(
            {
                "name": state,
                "group_type": "state",
                "count": metrics["count"],
                "acc": metrics["acc"],
                "weighted_f1": metrics["weighted_f1"],
                "scarcity_f1": per_class[0],
                "deficit_f1": per_class[1],
                "normal_f1": per_class[2],
                "excess_f1": per_class[3],
                "large_excess_f1": per_class[4],
            }
        )

    region_rows = []
    region_summary = {}
    for region in sorted(region_labels):
        metrics = compute_metrics(region_labels[region], region_preds[region], num_classes)
        region_summary[region] = metrics
        per_class = metrics["per_class_f1"]
        region_rows.append(
            {
                "name": region,
                "group_type": "region",
                "count": metrics["count"],
                "acc": metrics["acc"],
                "weighted_f1": metrics["weighted_f1"],
                "scarcity_f1": per_class[0],
                "deficit_f1": per_class[1],
                "normal_f1": per_class[2],
                "excess_f1": per_class[3],
                "large_excess_f1": per_class[4],
            }
        )

    write_metrics_csv(os.path.join(args.run_dir, "per_state_metrics.csv"), state_rows)
    write_metrics_csv(os.path.join(args.run_dir, "per_region_metrics.csv"), region_rows)

    summary = {
        "run_dir": args.run_dir,
        "dataset_dir": args.dataset_dir,
        "region_groups": REGION_GROUPS,
        "state_metrics": state_summary,
        "region_metrics": region_summary,
    }
    with open(os.path.join(args.run_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", os.path.join(args.run_dir, "per_state_metrics.csv"))
    print("Wrote:", os.path.join(args.run_dir, "per_region_metrics.csv"))
    print("Wrote:", os.path.join(args.run_dir, "analysis_summary.json"))


if __name__ == "__main__":
    main()
