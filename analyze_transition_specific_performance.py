#!/usr/bin/env python3
"""
Analyze monthly autoregressive model performance by seasonal transition.

This joins:
  - dataset-dir/samples.csv
  - run-dir/test_predictions.csv

and reports metrics grouped by:
  - transition_label
  - season_phase
  - target_month
  - state x transition_label

The output is designed for rebuttal tables: exact accuracy is reported, but
ordinal metrics are emphasized because rainfall classes are ordered.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CLASS_NAMES = ["scarcity", "deficit", "normal", "excess", "large_excess"]
MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze performance by seasonal transition")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more run directories containing test_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="baseline_runs/northwest_transition_analysis",
        help="Directory for combined transition analysis outputs",
    )
    return parser.parse_args()


def load_samples(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as f:
        rows = {}
        for row in csv.DictReader(f):
            rows[int(row["sample_id"])] = dict(row)
    return rows


def load_predictions(path: Path) -> list[dict[str, int]]:
    with path.open(newline="") as f:
        out = []
        for row in csv.DictReader(f):
            out.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "y_true": int(row["y_true"]),
                    "y_pred": int(row["y_pred"]),
                }
            )
    return out


def load_metrics(path: Path) -> dict:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open() as f:
        return json.load(f)


def model_label(run_dir: Path, metrics: dict) -> str:
    model = metrics.get("model", run_dir.name)
    loss = metrics.get("loss", "")
    tuned = ", tuned" if "optuna" in str(run_dir) else ""
    return f"{model} ({loss}{tuned})"


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int = 5) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        matrix[int(label), int(pred)] += 1
    return matrix


def safe_divide(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num / den)


def f1_scores(labels: np.ndarray, preds: np.ndarray, num_classes: int = 5) -> tuple[float, list[float]]:
    matrix = confusion_matrix(labels, preds, num_classes)
    supports = matrix.sum(axis=1)
    total = supports.sum()
    per_class = []
    for i in range(num_classes):
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - matrix[i, i])
        fn = float(matrix[i, :].sum() - matrix[i, i])
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        per_class.append(safe_divide(2 * precision * recall, precision + recall))
    weighted = safe_divide(float(np.sum(np.asarray(per_class) * supports)), float(total))
    return weighted, per_class


def group_metrics(labels: list[int], preds: list[int]) -> dict[str, object]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(preds, dtype=np.int64)
    errors = np.abs(p - y)
    signed = p - y
    matrix = confusion_matrix(y, p)
    weighted_f1, per_class_f1 = f1_scores(y, p)
    recalls = []
    supports = []
    for i in range(5):
        support = int(matrix[i].sum())
        supports.append(support)
        recalls.append(safe_divide(float(matrix[i, i]), float(support)))
    return {
        "count": int(y.size),
        "acc": float(np.mean(y == p)),
        "weighted_f1": weighted_f1,
        "within_1_accuracy": float(np.mean(errors <= 1)),
        "within_2_accuracy": float(np.mean(errors <= 2)),
        "mean_abs_class_error": float(errors.mean()),
        "severe_error_rate_ge2": float(np.mean(errors >= 2)),
        "opposite_extreme_rate": float(np.mean(((y == 0) & (p == 4)) | ((y == 4) & (p == 0)))),
        "mean_signed_error": float(signed.mean()),
        "scarcity_recall": recalls[0],
        "deficit_recall": recalls[1],
        "normal_recall": recalls[2],
        "excess_recall": recalls[3],
        "large_excess_recall": recalls[4],
        "scarcity_support": supports[0],
        "deficit_support": supports[1],
        "normal_support": supports[2],
        "excess_support": supports[3],
        "large_excess_support": supports[4],
        "scarcity_f1": per_class_f1[0],
        "deficit_f1": per_class_f1[1],
        "normal_f1": per_class_f1[2],
        "excess_f1": per_class_f1[3],
        "large_excess_f1": per_class_f1[4],
    }


def build_joined_rows(samples: dict[int, dict[str, str]], predictions: list[dict[str, int]]) -> list[dict[str, object]]:
    rows = []
    for pred in predictions:
        sample_id = pred["sample_id"]
        if sample_id not in samples:
            raise KeyError(f"sample_id {sample_id} missing from samples.csv")
        sample = samples[sample_id]
        row = dict(sample)
        row.update(pred)
        row["target_month"] = int(row["target_month"])
        row["target_month_name"] = MONTH_NAMES[int(row["target_month"])]
        rows.append(row)
    return rows


def summarize_group(rows: list[dict[str, object]], keys: list[str], model: str, run_dir: str) -> list[dict[str, object]]:
    grouped: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: {"labels": [], "preds": []})
    for row in rows:
        group_key = tuple(row[key] for key in keys)
        grouped[group_key]["labels"].append(int(row["y_true"]))
        grouped[group_key]["preds"].append(int(row["y_pred"]))

    out = []
    for group_key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        metric_row = group_metrics(values["labels"], values["preds"])
        base = {"model": model, "run_dir": run_dir}
        for key, value in zip(keys, group_key):
            base[key] = value
        base.update(metric_row)
        out.append(base)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_by_transition(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["transition_label"])].append(row)
    out = []
    for transition, transition_rows in sorted(grouped.items()):
        best_acc = max(transition_rows, key=lambda r: float(r["acc"]))
        best_wf1 = max(transition_rows, key=lambda r: float(r["weighted_f1"]))
        best_ordinal = max(transition_rows, key=lambda r: float(r["within_1_accuracy"]))
        out.append(
            {
                "transition_label": transition,
                "best_acc_model": best_acc["model"],
                "best_acc": best_acc["acc"],
                "best_wf1_model": best_wf1["model"],
                "best_wf1": best_wf1["weighted_f1"],
                "best_within1_model": best_ordinal["model"],
                "best_within1": best_ordinal["within_1_accuracy"],
                "lowest_severe_model": min(transition_rows, key=lambda r: float(r["severe_error_rate_ge2"]))["model"],
                "lowest_severe_rate": min(transition_rows, key=lambda r: float(r["severe_error_rate_ge2"]))["severe_error_rate_ge2"],
            }
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    samples = load_samples(Path(args.dataset_dir) / "samples.csv")

    transition_rows = []
    season_rows = []
    month_rows = []
    state_transition_rows = []
    run_summaries = []

    for run_dir_text in args.run_dirs:
        run_dir = Path(run_dir_text)
        metrics = load_metrics(run_dir)
        label = model_label(run_dir, metrics)
        predictions = load_predictions(run_dir / "test_predictions.csv")
        joined = build_joined_rows(samples, predictions)

        transition = summarize_group(joined, ["transition_label"], label, str(run_dir))
        season = summarize_group(joined, ["season_phase"], label, str(run_dir))
        month = summarize_group(joined, ["target_month", "target_month_name"], label, str(run_dir))
        state_transition = summarize_group(joined, ["state", "transition_label"], label, str(run_dir))

        transition_rows.extend(transition)
        season_rows.extend(season)
        month_rows.extend(month)
        state_transition_rows.extend(state_transition)

        all_metrics = group_metrics([int(r["y_true"]) for r in joined], [int(r["y_pred"]) for r in joined])
        run_summaries.append({"model": label, "run_dir": str(run_dir), **all_metrics})

        run_out = run_dir / "transition_analysis"
        write_csv(run_out / "by_transition.csv", transition)
        write_csv(run_out / "by_season_phase.csv", season)
        write_csv(run_out / "by_target_month.csv", month)
        write_csv(run_out / "by_state_transition.csv", state_transition)
        print(f"Wrote transition analysis for {run_dir}")

    write_csv(output_dir / "by_transition_all_models.csv", transition_rows)
    write_csv(output_dir / "by_season_phase_all_models.csv", season_rows)
    write_csv(output_dir / "by_target_month_all_models.csv", month_rows)
    write_csv(output_dir / "by_state_transition_all_models.csv", state_transition_rows)
    write_csv(output_dir / "run_summary.csv", run_summaries)
    write_csv(output_dir / "best_model_by_transition.csv", best_by_transition(transition_rows))

    print(f"Wrote combined transition analysis to {output_dir}")


if __name__ == "__main__":
    main()
