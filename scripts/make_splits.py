#!/usr/bin/env python3
"""
Create train/validation/test splits from the extracted patch dataset.

Inputs:
- patch_dataset/X.npy
- patch_dataset/y.npy
- patch_dataset/samples.csv

Outputs:
- train_idx.npy
- val_idx.npy
- test_idx.npy
- split_summary.csv
- split_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument(
        "--dataset-dir",
        default="patch_dataset",
        help="Directory containing X.npy, y.npy, and samples.csv",
    )
    parser.add_argument(
        "--split-mode",
        default="year_holdout",
        choices=["balanced_random", "year_holdout"],
        help=(
            "Split strategy. 'year_holdout' assigns entire years to train/val/test. "
            "'balanced_random' keeps the legacy random within-bucket split."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training split ratio for balanced_random mode",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio for balanced_random mode",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio for balanced_random mode",
    )
    parser.add_argument(
        "--train-years",
        default="2020,2021,2022",
        help="Comma-separated train years for year_holdout mode",
    )
    parser.add_argument(
        "--val-years",
        default="2023",
        help="Comma-separated validation years for year_holdout mode",
    )
    parser.add_argument(
        "--test-years",
        default="2024",
        help="Comma-separated test years for year_holdout mode",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=0,
        help="Optional cap on retained samples per split group; 0 keeps all samples",
    )
    parser.add_argument(
        "--min-per-group",
        type=int,
        default=1,
        help="Minimum required samples to keep a split group",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")


def compute_split_sizes(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        overflow = n_train + n_val - n
        n_val = max(0, n_val - overflow)
    n_test = n - n_train - n_val

    if n >= 3:
        if n_train == 0:
            n_train = 1
            n_test = max(0, n_test - 1)
        if n_val == 0:
            n_val = 1
            n_test = max(0, n_test - 1)
        if n_test == 0:
            n_test = 1
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_train = max(1, n_train - 1)
    return n_train, n_val, n_test


def load_samples(samples_path: str) -> list[dict[str, str]]:
    with open(samples_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"sample_id", "state", "label"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"samples.csv is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def validate_lengths(dataset_dir: str, samples: list[dict[str, str]]) -> None:
    x_path = os.path.join(dataset_dir, "X.npy")
    y_path = os.path.join(dataset_dir, "y.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(x_path)
    if not os.path.exists(y_path):
        raise FileNotFoundError(y_path)

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    if len(X) != len(y) or len(X) != len(samples):
        raise ValueError("X.npy, y.npy, and samples.csv have inconsistent lengths")


def assign_year_split(year: int, train_years: set[int], val_years: set[int], test_years: set[int]) -> str | None:
    if year in train_years:
        return "train"
    if year in val_years:
        return "val"
    if year in test_years:
        return "test"
    return None


def save_indices(dataset_dir: str, train_idx: list[int], val_idx: list[int], test_idx: list[int]) -> None:
    np.save(os.path.join(dataset_dir, "train_idx.npy"), np.array(sorted(train_idx), dtype=np.int64))
    np.save(os.path.join(dataset_dir, "val_idx.npy"), np.array(sorted(val_idx), dtype=np.int64))
    np.save(os.path.join(dataset_dir, "test_idx.npy"), np.array(sorted(test_idx), dtype=np.int64))


def write_summary(path: str, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "split",
        "state",
        "label",
        "year",
        "original_count",
        "kept_count",
        "train_count",
        "val_count",
        "test_count",
        "status",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    samples_path = os.path.join(dataset_dir, "samples.csv")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(samples_path)

    samples = load_samples(samples_path)
    validate_lengths(dataset_dir, samples)
    rng = np.random.default_rng(args.seed)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    summary_rows: list[dict[str, object]] = []

    if args.split_mode == "balanced_random":
        validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
        grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
        for row in samples:
            state = row["state"].strip()
            label = int(row["label"])
            sample_id = int(row["sample_id"])
            grouped[(state, label)].append(sample_id)

        for (state, label), sample_ids in sorted(grouped.items()):
            sample_ids = sample_ids.copy()
            bucket_size_original = len(sample_ids)

            if bucket_size_original < args.min_per_group:
                summary_rows.append(
                    {
                        "split": "mixed",
                        "state": state,
                        "label": label,
                        "year": "",
                        "original_count": bucket_size_original,
                        "kept_count": 0,
                        "train_count": 0,
                        "val_count": 0,
                        "test_count": 0,
                        "status": "dropped_min_threshold",
                    }
                )
                continue

            rng.shuffle(sample_ids)
            if args.max_per_group > 0:
                sample_ids = sample_ids[: args.max_per_group]

            kept_count = len(sample_ids)
            n_train, n_val, n_test = compute_split_sizes(
                kept_count, args.train_ratio, args.val_ratio
            )
            train_ids = sample_ids[:n_train]
            val_ids = sample_ids[n_train : n_train + n_val]
            test_ids = sample_ids[n_train + n_val : n_train + n_val + n_test]

            train_indices.extend(train_ids)
            val_indices.extend(val_ids)
            test_indices.extend(test_ids)

            summary_rows.append(
                {
                    "split": "mixed",
                    "state": state,
                    "label": label,
                    "year": "",
                    "original_count": bucket_size_original,
                    "kept_count": kept_count,
                    "train_count": len(train_ids),
                    "val_count": len(val_ids),
                    "test_count": len(test_ids),
                    "status": "kept",
                }
            )
    else:
        train_years = set(parse_int_list(args.train_years))
        val_years = set(parse_int_list(args.val_years))
        test_years = set(parse_int_list(args.test_years))
        all_years = train_years | val_years | test_years
        if not all_years:
            raise ValueError("At least one year must be provided across train/val/test")
        if (train_years & val_years) or (train_years & test_years) or (val_years & test_years):
            raise ValueError("train/val/test year sets must be disjoint")

        grouped: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
        skipped_without_year = 0

        for row in samples:
            year_text = row.get("year", "").strip()
            if not year_text:
                skipped_without_year += 1
                continue

            year = int(year_text)
            split = assign_year_split(year, train_years, val_years, test_years)
            if split is None:
                continue

            state = row["state"].strip()
            label = int(row["label"])
            sample_id = int(row["sample_id"])
            grouped[(split, state, label, year)].append(sample_id)

        if skipped_without_year and not grouped:
            raise ValueError(
                "samples.csv does not contain usable year metadata. "
                "Re-run scripts/extract_patches.py with --sample-mode year_forecast."
            )

        for (split, state, label, year), sample_ids in sorted(grouped.items()):
            sample_ids = sample_ids.copy()
            bucket_size_original = len(sample_ids)

            if bucket_size_original < args.min_per_group:
                summary_rows.append(
                    {
                        "split": split,
                        "state": state,
                        "label": label,
                        "year": year,
                        "original_count": bucket_size_original,
                        "kept_count": 0,
                        "train_count": 0,
                        "val_count": 0,
                        "test_count": 0,
                        "status": "dropped_min_threshold",
                    }
                )
                continue

            rng.shuffle(sample_ids)
            if args.max_per_group > 0:
                sample_ids = sample_ids[: args.max_per_group]
            kept_count = len(sample_ids)

            if split == "train":
                train_indices.extend(sample_ids)
                train_count, val_count, test_count = kept_count, 0, 0
            elif split == "val":
                val_indices.extend(sample_ids)
                train_count, val_count, test_count = 0, kept_count, 0
            else:
                test_indices.extend(sample_ids)
                train_count, val_count, test_count = 0, 0, kept_count

            summary_rows.append(
                {
                    "split": split,
                    "state": state,
                    "label": label,
                    "year": year,
                    "original_count": bucket_size_original,
                    "kept_count": kept_count,
                    "train_count": train_count,
                    "val_count": val_count,
                    "test_count": test_count,
                    "status": "kept",
                }
            )

    save_indices(dataset_dir, train_indices, val_indices, test_indices)
    write_summary(os.path.join(dataset_dir, "split_summary.csv"), summary_rows)

    config = {
        "dataset_dir": dataset_dir,
        "split_mode": args.split_mode,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "train_years": parse_int_list(args.train_years),
        "val_years": parse_int_list(args.val_years),
        "test_years": parse_int_list(args.test_years),
        "max_per_group": args.max_per_group,
        "min_per_group": args.min_per_group,
        "seed": args.seed,
        "num_train": int(len(train_indices)),
        "num_val": int(len(val_indices)),
        "num_test": int(len(test_indices)),
    }
    with open(os.path.join(dataset_dir, "split_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Split mode: {args.split_mode}")
    print(f"Train: {len(train_indices)}")
    print(f"Val:   {len(val_indices)}")
    print(f"Test:  {len(test_indices)}")
    print(f"Wrote summary: {os.path.join(dataset_dir, 'split_summary.csv')}")


if __name__ == "__main__":
    main()
