#!/usr/bin/env python3
"""
Analyze errors for extreme rainfall regimes.

Extreme classes:
- 0: Scarcity
- 4: Large Excess

The script joins test predictions with sample metadata and predictor patches,
then reports where extremes are correctly predicted or missed and how predictor
values differ between correct and missed extreme samples.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EXTREME_CLASSES = {
    0: "scarcity",
    4: "large_excess",
}

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
    parser = argparse.ArgumentParser(description="Analyze extreme-regime prediction errors")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to <run-dir>/extreme_error_analysis",
    )
    parser.add_argument(
        "--classes",
        default="0,4",
        help="Comma-separated class ids to analyze. Defaults to Scarcity and Large Excess.",
    )
    parser.add_argument(
        "--max-predictor-samples",
        type=int,
        default=50000,
        help="Optional cap for predictor-stat computation to keep analysis fast.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_class_ids(text: str) -> list[int]:
    class_ids = []
    for token in text.split(","):
        token = token.strip()
        if token:
            class_ids.append(int(token))
    if not class_ids:
        raise ValueError("Expected at least one class id")
    return class_ids


def region_for_state(state: str) -> str:
    for region, states in REGION_GROUPS.items():
        if state in states:
            return region
    return "other"


def load_samples(path: Path) -> dict[int, dict[str, str]]:
    rows = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[int(row["sample_id"])] = row
    return rows


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "y_true": int(row["y_true"]),
                    "y_pred": int(row["y_pred"]),
                }
            )
    return rows


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def summarize_group(rows: list[dict], group_key: str, class_ids: list[int]) -> list[dict]:
    grouped: dict[tuple[str, int], Counter] = defaultdict(Counter)
    for row in rows:
        y_true = row["y_true"]
        if y_true not in class_ids:
            continue
        key = (row[group_key], y_true)
        grouped[key]["support"] += 1
        if row["y_pred"] == y_true:
            grouped[key]["correct"] += 1
        else:
            grouped[key]["missed"] += 1
            grouped[key][f"pred_{row['y_pred']}"] += 1

    output = []
    for (group_value, class_id), counts in sorted(grouped.items()):
        support = int(counts["support"])
        correct = int(counts["correct"])
        missed = int(counts["missed"])
        most_common_wrong = ""
        most_common_wrong_count = 0
        for key, value in counts.items():
            if key.startswith("pred_") and value > most_common_wrong_count:
                most_common_wrong = key.replace("pred_", "")
                most_common_wrong_count = int(value)

        output.append(
            {
                group_key: group_value,
                "class_id": class_id,
                "class_name": EXTREME_CLASSES.get(class_id, f"class_{class_id}"),
                "support": support,
                "correct": correct,
                "missed": missed,
                "recall": safe_divide(correct, support),
                "miss_rate": safe_divide(missed, support),
                "most_common_wrong_class": most_common_wrong,
                "most_common_wrong_count": most_common_wrong_count,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def modality_from_channel(channel_name: str) -> str:
    if channel_name.startswith("soil_moisture"):
        return "soil_moisture"
    if channel_name.startswith("wind_speed"):
        return "wind_speed"
    if channel_name.startswith("relative_humidity"):
        return "rh"
    if channel_name.startswith("elevation"):
        return "elevation"
    return channel_name.split("_")[0]


def summarize_predictors(
    dataset_dir: Path,
    samples: dict[int, dict[str, str]],
    prediction_rows: list[dict],
    class_ids: list[int],
    max_samples: int,
    seed: int,
) -> list[dict]:
    config_path = dataset_dir / "config.json"
    x_path = dataset_dir / "X.npy"
    if not config_path.exists() or not x_path.exists():
        return []

    with config_path.open() as f:
        config = json.load(f)
    channel_names = config.get("channel_names", [])
    if not channel_names:
        return []

    modality_channels: dict[str, list[int]] = defaultdict(list)
    for idx, channel_name in enumerate(channel_names):
        modality_channels[modality_from_channel(channel_name)].append(idx)

    extreme_rows = [row for row in prediction_rows if row["y_true"] in class_ids]
    if max_samples > 0 and len(extreme_rows) > max_samples:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(extreme_rows), size=max_samples, replace=False)
        extreme_rows = [extreme_rows[int(idx)] for idx in selected]

    X = np.load(x_path, mmap_mode="r")
    buckets: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    for row in extreme_rows:
        sample = samples[row["sample_id"]]
        state = sample["state"]
        region = region_for_state(state)
        class_id = row["y_true"]
        status = "correct" if row["y_pred"] == row["y_true"] else "missed"
        patch = np.asarray(X[row["sample_id"]], dtype=np.float32)
        for modality, indices in modality_channels.items():
            value = float(np.nanmean(patch[indices]))
            buckets[(class_id, status, region, modality)].append(value)

    rows = []
    for (class_id, status, region, modality), values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "class_id": class_id,
                "class_name": EXTREME_CLASSES.get(class_id, f"class_{class_id}"),
                "status": status,
                "region": region,
                "modality": modality,
                "count": int(arr.size),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "median": float(np.median(arr)),
                "q25": float(np.quantile(arr, 0.25)),
                "q75": float(np.quantile(arr, 0.75)),
            }
        )
    return rows


def make_summary(
    prediction_rows: list[dict],
    class_ids: list[int],
    by_region_rows: list[dict],
    by_state_rows: list[dict],
) -> dict:
    summary = {"num_predictions": len(prediction_rows)}
    for class_id in class_ids:
        rows = [row for row in prediction_rows if row["y_true"] == class_id]
        support = len(rows)
        correct = sum(1 for row in rows if row["y_pred"] == row["y_true"])
        summary[f"{EXTREME_CLASSES.get(class_id, f'class_{class_id}')}_support"] = support
        summary[f"{EXTREME_CLASSES.get(class_id, f'class_{class_id}')}_recall"] = safe_divide(correct, support)

    for class_id in class_ids:
        region_candidates = [row for row in by_region_rows if row["class_id"] == class_id and row["support"] > 0]
        if region_candidates:
            lowest = min(region_candidates, key=lambda row: row["recall"])
            highest = max(region_candidates, key=lambda row: row["recall"])
            name = EXTREME_CLASSES.get(class_id, f"class_{class_id}")
            summary[f"{name}_lowest_recall_region"] = lowest["region"]
            summary[f"{name}_lowest_recall"] = lowest["recall"]
            summary[f"{name}_highest_recall_region"] = highest["region"]
            summary[f"{name}_highest_recall"] = highest["recall"]

        state_candidates = [row for row in by_state_rows if row["class_id"] == class_id and row["support"] >= 5]
        if state_candidates:
            lowest_state = min(state_candidates, key=lambda row: row["recall"])
            highest_state = max(state_candidates, key=lambda row: row["recall"])
            name = EXTREME_CLASSES.get(class_id, f"class_{class_id}")
            summary[f"{name}_lowest_recall_state"] = lowest_state["state"]
            summary[f"{name}_lowest_state_recall"] = lowest_state["recall"]
            summary[f"{name}_highest_recall_state"] = highest_state["state"]
            summary[f"{name}_highest_state_recall"] = highest_state["recall"]
    return summary


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "extreme_error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_ids = parse_class_ids(args.classes)
    samples = load_samples(dataset_dir / "samples.csv")
    prediction_rows = load_predictions(run_dir / "test_predictions.csv")

    joined_rows = []
    for row in prediction_rows:
        sample = samples[row["sample_id"]]
        state = sample["state"]
        joined_rows.append(
            {
                **row,
                "state": state,
                "region": region_for_state(state),
                "target_year": sample.get("target_year", ""),
                "row": sample.get("row", ""),
                "col": sample.get("col", ""),
            }
        )

    by_region = summarize_group(joined_rows, "region", class_ids)
    by_state = summarize_group(joined_rows, "state", class_ids)
    predictor_summary = summarize_predictors(
        dataset_dir=dataset_dir,
        samples=samples,
        prediction_rows=prediction_rows,
        class_ids=class_ids,
        max_samples=args.max_predictor_samples,
        seed=args.seed,
    )
    summary = make_summary(joined_rows, class_ids, by_region, by_state)

    write_csv(output_dir / "extreme_by_region.csv", by_region)
    write_csv(output_dir / "extreme_by_state.csv", by_state)
    write_csv(output_dir / "extreme_predictor_summary.csv", predictor_summary)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {output_dir / 'extreme_by_region.csv'}")
    print(f"Wrote {output_dir / 'extreme_by_state.csv'}")
    print(f"Wrote {output_dir / 'extreme_predictor_summary.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")

    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
