#!/usr/bin/env python3
"""
Summarize region-specific modality ablations from existing model runs.

Expected inputs are run directories that already contain per_region_metrics.csv,
typically produced by:

python scripts/analyze_patch_predictions.py --dataset-dir <dataset> --run-dir <run_dir>

The script compares each ablation against a baseline run and reports the
region-wise change in weighted F1 and accuracy.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


MODALITY_ALIASES = {
    "no_rh": "No RH",
    "no_wind": "No Wind",
    "no_wind_speed": "No Wind",
    "no_soil": "No Soil Moisture",
    "no_soil_moisture": "No Soil Moisture",
    "no_elevation": "No Elevation",
    "no_ndvi": "No NDVI",
    "no_lst": "No LST",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize region-specific modality ablations")
    parser.add_argument("--dataset-dir", default="patch_dataset_monsoon_to_next_annual")
    parser.add_argument("--baseline-run", required=True, help="Run directory for the full-modality baseline")
    parser.add_argument("--ablation-runs", nargs="+", required=True, help="Run directories for ablations")
    parser.add_argument("--output-csv", default="baseline_runs/region_modality_ablation_summary.csv")
    parser.add_argument("--output-latex", default="baseline_runs/region_modality_ablation_table.tex")
    parser.add_argument("--ensure-analysis", action="store_true", help="Run scripts/analyze_patch_predictions.py if needed")
    parser.add_argument("--sort-by-region", default="", help="Optional region name to sort rows by W-F1 drop")
    return parser.parse_args()


def infer_modality_name(run_dir: Path) -> str:
    name = run_dir.name.lower()
    for token, label in MODALITY_ALIASES.items():
        if token in name:
            return label
    return run_dir.name


def ensure_region_metrics(dataset_dir: Path, run_dir: Path, ensure_analysis: bool) -> None:
    metrics_path = run_dir / "per_region_metrics.csv"
    if metrics_path.exists():
        return
    if not ensure_analysis:
        raise FileNotFoundError(
            f"Missing {metrics_path}. Re-run with --ensure-analysis or run scripts/analyze_patch_predictions.py first."
        )
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_patch_predictions.py",
            "--dataset-dir",
            str(dataset_dir),
            "--run-dir",
            str(run_dir),
        ],
        check=True,
    )


def read_region_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    path = run_dir / "per_region_metrics.csv"
    rows = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            region = row["name"]
            rows[region] = {
                "count": float(row["count"]),
                "acc": float(row["acc"]),
                "weighted_f1": float(row["weighted_f1"]),
                "scarcity_f1": float(row["scarcity_f1"]),
                "deficit_f1": float(row["deficit_f1"]),
                "normal_f1": float(row["normal_f1"]),
                "excess_f1": float(row["excess_f1"]),
                "large_excess_f1": float(row["large_excess_f1"]),
            }
    return rows


def write_summary_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "modality",
        "region",
        "count",
        "baseline_acc",
        "ablation_acc",
        "delta_acc",
        "baseline_wf1",
        "ablation_wf1",
        "delta_wf1",
        "baseline_large_excess_f1",
        "ablation_large_excess_f1",
        "delta_large_excess_f1",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def write_latex(rows: list[dict], output_latex: Path) -> None:
    output_latex.parent.mkdir(parents=True, exist_ok=True)
    with output_latex.open("w") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(
            "\\caption{Region-specific leave-one-modality-out ablation. "
            "Negative $\\Delta$ W-F1 means the region depends on the dropped modality.}\n"
        )
        f.write("\\label{tab:region_modality_ablation}\n")
        f.write("\\begin{tabular}{l l c c c c}\n")
        f.write("\\hline\n")
        f.write("Dropped Modality & Region & Base W-F1 & Ablated W-F1 & $\\Delta$ W-F1 & $\\Delta$ Large Excess \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(
                f"{row['modality']} & "
                f"{row['region'].replace('_', ' ').title()} & "
                f"{format_float(float(row['baseline_wf1']))} & "
                f"{format_float(float(row['ablation_wf1']))} & "
                f"{format_float(float(row['delta_wf1']))} & "
                f"{format_float(float(row['delta_large_excess_f1']))} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    baseline_run = Path(args.baseline_run)
    ablation_runs = [Path(path) for path in args.ablation_runs]

    ensure_region_metrics(dataset_dir, baseline_run, args.ensure_analysis)
    for run_dir in ablation_runs:
        ensure_region_metrics(dataset_dir, run_dir, args.ensure_analysis)

    baseline = read_region_metrics(baseline_run)
    rows = []
    for run_dir in ablation_runs:
        modality = infer_modality_name(run_dir)
        ablation = read_region_metrics(run_dir)
        for region, base_values in baseline.items():
            if region not in ablation:
                continue
            ablation_values = ablation[region]
            rows.append(
                {
                    "modality": modality,
                    "region": region,
                    "count": int(base_values["count"]),
                    "baseline_acc": base_values["acc"],
                    "ablation_acc": ablation_values["acc"],
                    "delta_acc": ablation_values["acc"] - base_values["acc"],
                    "baseline_wf1": base_values["weighted_f1"],
                    "ablation_wf1": ablation_values["weighted_f1"],
                    "delta_wf1": ablation_values["weighted_f1"] - base_values["weighted_f1"],
                    "baseline_large_excess_f1": base_values["large_excess_f1"],
                    "ablation_large_excess_f1": ablation_values["large_excess_f1"],
                    "delta_large_excess_f1": ablation_values["large_excess_f1"] - base_values["large_excess_f1"],
                }
            )

    if args.sort_by_region:
        rows.sort(
            key=lambda row: (
                row["region"] != args.sort_by_region,
                float(row["delta_wf1"]),
                row["modality"],
            )
        )
    else:
        rows.sort(key=lambda row: (row["region"], float(row["delta_wf1"]), row["modality"]))

    output_csv = Path(args.output_csv)
    output_latex = Path(args.output_latex)
    write_summary_csv(rows, output_csv)
    write_latex(rows, output_latex)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_latex}")

    print("\nLargest W-F1 drops by region:")
    regions = sorted({row["region"] for row in rows})
    for region in regions:
        region_rows = [row for row in rows if row["region"] == region]
        worst = min(region_rows, key=lambda row: float(row["delta_wf1"]))
        print(
            f"{region}: {worst['modality']} "
            f"delta_wf1={float(worst['delta_wf1']):.4f} "
            f"({float(worst['baseline_wf1']):.4f}->{float(worst['ablation_wf1']):.4f})"
        )


if __name__ == "__main__":
    main()
