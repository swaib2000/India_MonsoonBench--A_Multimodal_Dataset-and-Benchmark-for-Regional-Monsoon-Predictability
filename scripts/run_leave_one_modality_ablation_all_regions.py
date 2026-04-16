#!/usr/bin/env python3
"""
Run leave-one-out modality ablations for the regional monthly AR benchmark.

For each region, this script:
1. Selects the best completed model among Conv3D / ConvLSTM / Swin3D.
2. Retrains that model while zeroing one predictor modality at a time.
3. Runs the standard prediction, ordinal-distance, and extreme-regime analyses.
4. Writes one combined CSV and LaTeX table for rebuttal reporting.

The script is resumable: if a run directory already contains metrics.json, the
training step is skipped unless --force is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_LR = 1.5591367900525532e-4
DEFAULT_WEIGHT_DECAY = 3.6315125458268e-5

REGIONS: dict[str, dict[str, Any]] = {
    "northwest_himalayan": {
        "label": "Northwest-Himalayan",
        "dataset_dir": "patch_dataset_monthly_ar_northwest_himalayan_full",
        "baseline_runs": {
            "conv3d": "baseline_runs/optuna_northwest_monthly_ar/trial_0023",
            "convlstm": "baseline_runs/convlstm_monthly_ar_northwest_himalayan_tuned",
            "swin3d": "baseline_runs/swin3d_monthly_ar_northwest_himalayan_tuned",
        },
    },
    "central_monsoon_core": {
        "label": "Central Indian Monsoon Core",
        "dataset_dir": "patch_dataset_monthly_ar_central_monsoon_core_full_bihar",
        "baseline_runs": {
            "conv3d": "baseline_runs/conv3d_monthly_ar_central_monsoon_core_full_bihar_transfer_ce",
            "convlstm": "baseline_runs/convlstm_monthly_ar_central_monsoon_core_full_bihar_transfer_ce",
            "swin3d": "baseline_runs/swin3d_monthly_ar_central_monsoon_core_full_bihar_transfer_ce",
        },
    },
    "south_peninsular_deccan": {
        "label": "South Peninsular-Deccan",
        "dataset_dir": "patch_dataset_monthly_ar_south_peninsular_deccan_full",
        "baseline_runs": {
            "conv3d": "baseline_runs/conv3d_monthly_ar_south_peninsular_deccan_transfer_ce",
            "convlstm": "baseline_runs/convlstm_monthly_ar_south_peninsular_deccan_transfer_ce",
            "swin3d": "baseline_runs/swin3d_monthly_ar_south_peninsular_deccan_transfer_ce",
        },
    },
    "east_northeast_humid_orographic": {
        "label": "East-Northeast Humid Orographic",
        "dataset_dir": "patch_dataset_monthly_ar_east_northeast_humid_orographic_full_assam",
        "baseline_runs": {
            "conv3d": "baseline_runs/conv3d_monthly_ar_east_northeast_humid_orographic_full_assam_transfer_ce",
            "convlstm": "baseline_runs/convlstm_monthly_ar_east_northeast_humid_orographic_full_assam_transfer_ce",
            "swin3d": "baseline_runs/swin3d_monthly_ar_east_northeast_humid_orographic_full_assam_transfer_ce",
        },
    },
}

MODALITIES = [
    "precipitation",
    "lst",
    "ndvi",
    "rh",
    "soil_moisture",
    "wind_speed",
    "elevation",
]

MODALITY_LABELS = {
    "precipitation": "Precipitation history",
    "lst": "Land surface temperature",
    "ndvi": "NDVI",
    "rh": "Relative humidity",
    "soil_moisture": "Soil moisture",
    "wind_speed": "Wind speed",
    "elevation": "Elevation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leave-one-out modality ablations for all regional monthly AR datasets."
    )
    parser.add_argument(
        "--regions",
        default=",".join(REGIONS),
        help=f"Comma-separated regions. Choices: {', '.join(REGIONS)}",
    )
    parser.add_argument(
        "--modalities",
        default=",".join(MODALITIES),
        help=f"Comma-separated modalities to ablate. Choices: {', '.join(MODALITIES)}",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["weighted_f1", "acc"],
        default="weighted_f1",
        help="Metric used to choose the best existing model per region.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "conv3d", "convlstm", "swin3d"],
        help="Use one fixed model for every region, or auto-select the best completed model.",
    )
    parser.add_argument("--output-root", default="baseline_runs/regional_modality_ablation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--batch-size-conv3d", type=int, default=32)
    parser.add_argument("--batch-size-convlstm", type=int, default=16)
    parser.add_argument("--batch-size-swin3d", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stats-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--loss", default="ce", choices=["ce", "focal", "ordinal", "ce_ordinal", "focal_ordinal"])
    parser.add_argument("--ordinal-weight", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--force", action="store_true", help="Retrain even if metrics.json already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="Skip missing baseline candidates instead of failing.",
    )
    return parser.parse_args()


def split_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def metric_value(metrics: dict[str, Any], selection_metric: str) -> float:
    key = "test_weighted_f1" if selection_metric == "weighted_f1" else "test_acc"
    value = metrics.get(key)
    if value is None or value == "":
        return float("-inf")
    return float(value)


def select_model(region_key: str, region_cfg: dict[str, Any], args: argparse.Namespace) -> tuple[str, Path, dict[str, Any]]:
    candidates = region_cfg["baseline_runs"]
    if args.model != "auto":
        run_dir = Path(candidates[args.model])
        metrics = read_json(run_dir / "metrics.json")
        if not metrics and not args.allow_missing_baseline:
            raise FileNotFoundError(f"Missing baseline metrics for {region_key}/{args.model}: {run_dir / 'metrics.json'}")
        return args.model, run_dir, metrics

    best_model = ""
    best_run_dir = Path()
    best_metrics: dict[str, Any] = {}
    best_value = float("-inf")

    for model_name, run_dir_text in candidates.items():
        run_dir = Path(run_dir_text)
        metrics = read_json(run_dir / "metrics.json")
        if not metrics:
            if args.allow_missing_baseline:
                continue
            raise FileNotFoundError(f"Missing baseline metrics for {region_key}/{model_name}: {run_dir / 'metrics.json'}")
        value = metric_value(metrics, args.selection_metric)
        if value > best_value:
            best_model = model_name
            best_run_dir = run_dir
            best_metrics = metrics
            best_value = value

    if not best_model:
        raise RuntimeError(f"No usable baseline runs found for {region_key}")
    return best_model, best_run_dir, best_metrics


def batch_size_for(model_name: str, args: argparse.Namespace) -> int:
    return {
        "conv3d": args.batch_size_conv3d,
        "convlstm": args.batch_size_convlstm,
        "swin3d": args.batch_size_swin3d,
    }[model_name]


def run_command(cmd: list[str], log_path: Path | None, dry_run: bool) -> None:
    printable = " ".join(cmd)
    if dry_run:
        print(f"[dry-run] {printable}")
        return

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log_file:
            print(f"$ {printable}", file=log_file)
            log_file.flush()
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)


def train_ablation(
    region_key: str,
    region_cfg: dict[str, Any],
    model_name: str,
    modality: str,
    run_dir: Path,
    args: argparse.Namespace,
) -> None:
    if (run_dir / "metrics.json").exists() and not args.force:
        print(f"Skipping existing training: {run_dir}")
        return

    cmd = [
        sys.executable,
        "scripts/train_patch_baselines.py",
        "--dataset-dir",
        region_cfg["dataset_dir"],
        "--output-dir",
        str(run_dir),
        "--model",
        model_name,
        "--loss",
        args.loss,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(batch_size_for(model_name, args)),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--stats-samples",
        str(args.stats_samples),
        "--seed",
        str(args.seed),
        "--focal-gamma",
        str(args.focal_gamma),
        "--ordinal-weight",
        str(args.ordinal_weight),
        "--drop-modalities",
        modality,
        "--disable-early-stopping",
    ]
    log_path = run_dir / f"train_drop_{modality}.log"
    print(f"Training {region_key}: {model_name}, no {modality}")
    run_command(cmd, log_path, args.dry_run)


def analyze_run(dataset_dir: str, run_dir: Path, args: argparse.Namespace) -> None:
    if args.dry_run:
        prefix = "[dry-run] "
    else:
        prefix = ""

    if not args.dry_run and not (run_dir / "metrics.json").exists():
        raise FileNotFoundError(f"Cannot analyze missing run: {run_dir / 'metrics.json'}")

    commands = [
        [
            sys.executable,
            "scripts/analyze_patch_predictions.py",
            "--dataset-dir",
            dataset_dir,
            "--run-dir",
            str(run_dir),
        ],
        [
            sys.executable,
            "scripts/analyze_extreme_regime_errors.py",
            "--dataset-dir",
            dataset_dir,
            "--run-dir",
            str(run_dir),
        ],
        [
            sys.executable,
            "scripts/compute_ordinal_metrics.py",
            "--run-dirs",
            str(run_dir),
            "--output-csv",
            str(run_dir / "ordinal_metrics_summary.csv"),
            "--write-json",
        ],
    ]

    for cmd in commands:
        print(f"{prefix}Analyzing: {' '.join(cmd)}")
        run_command(cmd, None, args.dry_run)


def summarize_run(
    region_key: str,
    region_label: str,
    dataset_dir: str,
    model_name: str,
    modality: str,
    run_dir: Path,
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    ordinal = read_json(run_dir / "ordinal_metrics.json")
    extreme = read_json(run_dir / "extreme_error_analysis" / "summary.json")

    baseline_acc = float(baseline_metrics.get("test_acc", 0.0) or 0.0)
    baseline_wf1 = float(baseline_metrics.get("test_weighted_f1", 0.0) or 0.0)
    test_acc = float(metrics.get("test_acc", 0.0) or 0.0)
    test_wf1 = float(metrics.get("test_weighted_f1", 0.0) or 0.0)

    return {
        "region": region_key,
        "region_label": region_label,
        "dataset_dir": dataset_dir,
        "model": model_name,
        "ablation": "full" if modality == "full" else f"without_{modality}",
        "dropped_modality": "" if modality == "full" else modality,
        "dropped_modality_label": "" if modality == "full" else MODALITY_LABELS.get(modality, modality),
        "run_dir": str(run_dir),
        "test_acc": test_acc,
        "test_weighted_f1": test_wf1,
        "delta_acc_vs_full": test_acc - baseline_acc if modality != "full" else 0.0,
        "delta_weighted_f1_vs_full": test_wf1 - baseline_wf1 if modality != "full" else 0.0,
        "within_1_accuracy": ordinal.get("within_1_accuracy", ""),
        "mean_abs_class_error": ordinal.get("mean_abs_class_error", ""),
        "severe_error_rate_ge2": ordinal.get("severe_error_rate_ge2", ""),
        "scarcity_recall": extreme.get("scarcity_recall", ordinal.get("scarcity_recall", "")),
        "large_excess_recall": extreme.get("large_excess_recall", ordinal.get("large_excess_recall", "")),
        "num_train": metrics.get("num_train", ""),
        "num_val": metrics.get("num_val", ""),
        "num_test": metrics.get("num_test", ""),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return "--"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def write_latex(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Leave-one-out modality ablation for regional monthly autoregressive rainfall forecasting. Negative $\\Delta$ indicates performance drop after removing the modality.}",
        "\\label{tab:regional_modality_ablation}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Region & Ablation & Acc. & $\\Delta$ Acc. & wF1 & $\\Delta$ wF1 \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["ablation"] == "full":
            ablation = "Full"
        else:
            ablation = f"w/o {row['dropped_modality_label']}"
        lines.append(
            f"{row['region_label']} & {ablation} & "
            f"{fmt(row['test_acc'])} & {fmt(row['delta_acc_vs_full'])} & "
            f"{fmt(row['test_weighted_f1'])} & {fmt(row['delta_weighted_f1_vs_full'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    selected_regions = split_csv(args.regions)
    selected_modalities = split_csv(args.modalities)

    unknown_regions = sorted(set(selected_regions) - set(REGIONS))
    unknown_modalities = sorted(set(selected_modalities) - set(MODALITIES))
    if unknown_regions:
        raise ValueError(f"Unknown regions: {', '.join(unknown_regions)}")
    if unknown_modalities:
        raise ValueError(f"Unknown modalities: {', '.join(unknown_modalities)}")

    output_root = Path(args.output_root)
    all_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for region_key in selected_regions:
        region_cfg = REGIONS[region_key]
        dataset_dir = region_cfg["dataset_dir"]
        model_name, baseline_run_dir, baseline_metrics = select_model(region_key, region_cfg, args)

        print(
            f"\nSelected {model_name} for {region_key} "
            f"({args.selection_metric}: {metric_value(baseline_metrics, args.selection_metric):.4f})"
        )
        selection_rows.append(
            {
                "region": region_key,
                "region_label": region_cfg["label"],
                "selected_model": model_name,
                "selection_metric": args.selection_metric,
                "selected_run_dir": str(baseline_run_dir),
                "baseline_test_acc": baseline_metrics.get("test_acc", ""),
                "baseline_test_weighted_f1": baseline_metrics.get("test_weighted_f1", ""),
            }
        )

        # Baseline/full row from the already completed best model.
        all_rows.append(
            summarize_run(
                region_key,
                region_cfg["label"],
                dataset_dir,
                model_name,
                "full",
                baseline_run_dir,
                baseline_metrics,
            )
        )

        for modality in selected_modalities:
            run_dir = output_root / region_key / f"{model_name}_without_{modality}"
            train_ablation(region_key, region_cfg, model_name, modality, run_dir, args)
            analyze_run(dataset_dir, run_dir, args)
            if not args.dry_run:
                all_rows.append(
                    summarize_run(
                        region_key,
                        region_cfg["label"],
                        dataset_dir,
                        model_name,
                        modality,
                        run_dir,
                        baseline_metrics,
                    )
                )

    if args.dry_run:
        print("\nDry run complete. No files were written.")
        return

    summary_csv = output_root / "leave_one_modality_ablation_summary.csv"
    selection_csv = output_root / "selected_models.csv"
    latex_path = output_root / "leave_one_modality_ablation_table.tex"
    write_csv(selection_rows, selection_csv)
    write_csv(all_rows, summary_csv)
    write_latex(all_rows, latex_path)

    print("\nLeave-one-out modality ablation complete.")
    print(f"Selected models: {selection_csv}")
    print(f"Summary CSV:     {summary_csv}")
    print(f"LaTeX table:     {latex_path}")


if __name__ == "__main__":
    main()
