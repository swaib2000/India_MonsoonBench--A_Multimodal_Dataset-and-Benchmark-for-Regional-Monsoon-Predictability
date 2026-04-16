#!/usr/bin/env python3
"""
Run season-ablation experiments for the year-sequence rainfall benchmark.

The script orchestrates:
1. Patch extraction for selected month groups.
2. Strict target-year train/val/test split.
3. Model training.
4. Per-state and per-region analysis.

It is intentionally a thin runner around the existing project scripts so every
ablation uses the same data format, split logic, losses, and metrics.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


PREDICTORS = ["lst", "ndvi", "rh", "soil_moisture", "wind_speed"]
SEASONS = {
    "winter_jf": [1, 2],
    "premonsoon_mam": [3, 4, 5],
    "monsoon_jjas": [6, 7, 8, 9],
    "postmonsoon_ond": [10, 11, 12],
}
FULL_YEAR = list(range(1, 13))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run season ablations for year-month forecasting")
    parser.add_argument("--manifest", default="modeling_manifest_full_year_states.csv")
    parser.add_argument("--source-manifest", default="modeling_manifest.csv")
    parser.add_argument("--dataset-root", default=".")
    parser.add_argument("--run-root", default="baseline_runs")
    parser.add_argument("--model", default="swin3d_yearmonth", choices=["swin3d_yearmonth", "conv3d", "swin3d"])
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--max-per-state", type=int, default=1000)
    parser.add_argument("--history-years", type=int, default=2)
    parser.add_argument("--target-year-offset", type=int, default=1)
    parser.add_argument("--years", default="2020,2021,2022,2023,2024")
    parser.add_argument("--train-years", default="2022")
    parser.add_argument("--val-years", default="2023")
    parser.add_argument("--test-years", default="2024")
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument(
        "--ablation-mode",
        default="both",
        choices=["only", "drop", "both"],
        help="'only' trains on one season; 'drop' removes one season from full year; 'both' runs both.",
    )
    parser.add_argument(
        "--include-full",
        action="store_true",
        help="Also train the full Jan-Dec reference run.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated variant names to run, e.g. monsoon_jjas,drop_monsoon_jjas.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], log_path: Path | None, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return

    if log_path is None:
        subprocess.run(cmd, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def ensure_full_year_manifest(source_manifest: Path, output_manifest: Path) -> None:
    if output_manifest.exists():
        return

    try:
        import rasterio
    except Exception as exc:  # pragma: no cover
        raise SystemExit("rasterio is required to auto-create the full-year manifest") from exc

    full_rows = []
    with source_manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Could not read fieldnames from {source_manifest}")
        for row in reader:
            keep = True
            for predictor in PREDICTORS:
                with rasterio.open(row[predictor]) as src:
                    if src.count != 60:
                        keep = False
                        break
            if keep:
                full_rows.append(row)

    with output_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(full_rows)
    print(f"Wrote {output_manifest} with {len(full_rows)} full-year states", flush=True)


def make_variants(mode: str, include_full: bool) -> dict[str, list[int]]:
    variants: dict[str, list[int]] = {}
    if include_full:
        variants["full_year"] = FULL_YEAR
    if mode in {"only", "both"}:
        variants.update(SEASONS)
    if mode in {"drop", "both"}:
        for name, months in SEASONS.items():
            variants[f"drop_{name}"] = [month for month in FULL_YEAR if month not in months]
    return variants


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    dataset_root = Path(args.dataset_root)
    run_root = Path(args.run_root)
    manifest = Path(args.manifest)
    source_manifest = Path(args.source_manifest)

    ensure_full_year_manifest(source_manifest, manifest)

    variants = make_variants(args.ablation_mode, args.include_full)
    if args.variants.strip():
        requested = {token.strip() for token in args.variants.split(",") if token.strip()}
        unknown = requested - set(variants)
        if unknown:
            raise ValueError(f"Unknown variants: {', '.join(sorted(unknown))}")
        variants = {name: months for name, months in variants.items() if name in requested}

    if not variants:
        raise SystemExit("No variants selected")

    for variant_name, months in variants.items():
        month_text = ",".join(str(month) for month in months)
        dataset_dir = dataset_root / f"patch_dataset_season_{variant_name}_2yr_next_annual_cap{args.max_per_state}"
        run_dir = run_root / f"{args.model}_season_{variant_name}_2yr_next_annual_cap{args.max_per_state}_{args.loss}"

        if args.overwrite or not (dataset_dir / "config.json").exists():
            run_command(
                [
                    sys.executable,
                    "-u",
                    "extract_patches.py",
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(dataset_dir),
                    "--sample-mode",
                    "year_sequence_forecast",
                    "--schema",
                    "monsoon_5yr",
                    "--years",
                    args.years,
                    "--input-months",
                    month_text,
                    "--target-period",
                    "annual",
                    "--target-year-offset",
                    str(args.target_year_offset),
                    "--history-years",
                    str(args.history_years),
                    "--patch-size",
                    str(args.patch_size),
                    "--stride",
                    str(args.stride),
                    "--max-per-state",
                    str(args.max_per_state),
                    "--min-valid-fraction",
                    str(args.min_valid_fraction),
                    "--fill-value",
                    str(args.fill_value),
                ],
                log_path=run_dir / f"extract_{variant_name}.log",
                dry_run=args.dry_run,
            )
        else:
            print(f"Skipping extraction for {variant_name}; found {dataset_dir / 'config.json'}", flush=True)

        if args.overwrite or not (dataset_dir / "split_config.json").exists():
            run_command(
                [
                    sys.executable,
                    "make_splits.py",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--split-mode",
                    "year_holdout",
                    "--train-years",
                    args.train_years,
                    "--val-years",
                    args.val_years,
                    "--test-years",
                    args.test_years,
                ],
                log_path=run_dir / f"split_{variant_name}.log",
                dry_run=args.dry_run,
            )
        else:
            print(f"Skipping split for {variant_name}; found {dataset_dir / 'split_config.json'}", flush=True)

        if args.overwrite or not (run_dir / "metrics.json").exists():
            run_command(
                [
                    sys.executable,
                    "-u",
                    "train_patch_baselines.py",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--output-dir",
                    str(run_dir),
                    "--model",
                    args.model,
                    "--loss",
                    args.loss,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                ],
                log_path=run_dir / f"train_{variant_name}.log",
                dry_run=args.dry_run,
            )
        else:
            print(f"Skipping training for {variant_name}; found {run_dir / 'metrics.json'}", flush=True)

        if not args.dry_run:
            run_command(
                [
                    sys.executable,
                    "analyze_patch_predictions.py",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--run-dir",
                    str(run_dir),
                ],
                log_path=run_dir / f"analyze_{variant_name}.log",
                dry_run=False,
            )

    print("\nSeason ablation complete.", flush=True)


if __name__ == "__main__":
    main()
