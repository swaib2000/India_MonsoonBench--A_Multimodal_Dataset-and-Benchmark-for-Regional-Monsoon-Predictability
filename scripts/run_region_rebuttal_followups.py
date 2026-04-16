#!/usr/bin/env python3
"""
Generic regional follow-up runner for monthly autoregressive rebuttal runs.

It waits for an Optuna search to finish, picks the best validation trial, runs
analysis for that trial, trains follow-up architectures with the best tuned
hyperparameters, analyzes them, computes ordinal metrics, transition-specific
metrics, and writes compact CSV/LaTeX summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regional rebuttal follow-up experiments")
    parser.add_argument("--region-name", required=True, help="Short name for output files, e.g. south_peninsular_deccan")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--optuna-root", required=True)
    parser.add_argument("--expected-trials", type=int, default=24)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stats-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--models", default="convlstm,swin3d")
    parser.add_argument("--disable-early-stopping", action="store_true", default=True)
    return parser.parse_args()


def run(cmd: list[str], log_path: Path | None = None) -> None:
    print("\n$", " ".join(cmd), flush=True)
    if log_path is None:
        completed = subprocess.run(cmd, check=False)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            completed = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode != 0:
        tail = ""
        if log_path and log_path.exists():
            tail = "\n".join(log_path.read_text().splitlines()[-40:])
        raise RuntimeError(f"Command failed with code {completed.returncode}.\n{tail}")


def read_leaderboard(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def complete_trials(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("state") == "COMPLETE" and row.get("best_val_acc")]


def wait_for_optuna(leaderboard: Path, expected_trials: int, poll_seconds: int) -> list[dict[str, str]]:
    while True:
        complete = complete_trials(read_leaderboard(leaderboard))
        print(f"Optuna progress: {len(complete)}/{expected_trials} complete", flush=True)
        if len(complete) >= expected_trials:
            return complete
        time.sleep(poll_seconds)


def best_trial(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        raise ValueError("No complete Optuna trials found")
    return max(rows, key=lambda row: float(row["best_val_acc"]))


def get_param(row: dict[str, str], key: str, default: str) -> str:
    value = row.get(key, "")
    return value if value not in {"", None} else default


def train_followup_model(args: argparse.Namespace, model: str, best: dict[str, str]) -> Path:
    batch_size = get_param(best, "batch_size", "16")
    if model in {"convlstm", "swin3d"} and int(float(batch_size)) > 16:
        batch_size = "16"

    run_dir = Path("baseline_runs") / f"{model}_monthly_ar_{args.region_name}_tuned"
    log_name = Path(f"train_{model}_monthly_ar_{args.region_name}_tuned.log")
    cmd = [
        sys.executable,
        "scripts/train_patch_baselines.py",
        "--dataset-dir",
        args.dataset_dir,
        "--output-dir",
        str(run_dir),
        "--model",
        model,
        "--loss",
        get_param(best, "loss", "ce"),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        batch_size,
        "--lr",
        get_param(best, "lr", "3e-4"),
        "--weight-decay",
        get_param(best, "weight_decay", "1e-3"),
        "--num-workers",
        str(args.num_workers),
        "--stats-samples",
        str(args.stats_samples),
        "--seed",
        str(args.seed),
        "--focal-gamma",
        get_param(best, "focal_gamma", "2.0"),
        "--ordinal-weight",
        get_param(best, "ordinal_weight", "0.3"),
    ]
    if args.disable_early_stopping:
        cmd.append("--disable-early-stopping")
    run(cmd, log_name)
    return run_dir


def analyze_run(dataset_dir: str, run_dir: Path) -> None:
    run([sys.executable, "scripts/analyze_patch_predictions.py", "--dataset-dir", dataset_dir, "--run-dir", str(run_dir)])
    run([sys.executable, "scripts/analyze_extreme_regime_errors.py", "--dataset-dir", dataset_dir, "--run-dir", str(run_dir)])


def main() -> None:
    args = parse_args()
    optuna_root = Path(args.optuna_root)
    leaderboard = optuna_root / "leaderboard.csv"
    complete = complete_trials(read_leaderboard(leaderboard)) if args.skip_wait else wait_for_optuna(
        leaderboard, args.expected_trials, args.poll_seconds
    )
    best = best_trial(complete)
    best_run_dir = Path(best["output_dir"])

    summary = {
        "region_name": args.region_name,
        "best_optuna_trial": best.get("number"),
        "best_optuna_run_dir": str(best_run_dir),
        "best_val_acc": best.get("best_val_acc"),
        "test_acc": best.get("test_acc"),
        "test_weighted_f1": best.get("test_weighted_f1"),
        "params": {key: best.get(key, "") for key in ["loss", "batch_size", "lr", "weight_decay", "focal_gamma", "ordinal_weight"]},
    }
    (optuna_root / "selected_best_trial.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    analyze_run(args.dataset_dir, best_run_dir)
    run_dirs = [str(best_run_dir)]
    for model in [token.strip() for token in args.models.split(",") if token.strip()]:
        run_dir = train_followup_model(args, model, best)
        analyze_run(args.dataset_dir, run_dir)
        run_dirs.append(str(run_dir))

    prefix = f"baseline_runs/{args.region_name}_monthly_ar"
    run(
        [
            sys.executable,
            "scripts/compute_ordinal_metrics.py",
            "--run-dirs",
            *run_dirs,
            "--output-csv",
            f"{prefix}_ordinal_metrics.csv",
            "--write-json",
        ]
    )
    run(
        [
            sys.executable,
            "scripts/analyze_transition_specific_performance.py",
            "--dataset-dir",
            args.dataset_dir,
            "--run-dirs",
            *run_dirs,
            "--output-dir",
            f"baseline_runs/{args.region_name}_transition_analysis",
        ]
    )
    run(
        [
            sys.executable,
            "scripts/collect_rebuttal_model_results.py",
            "--run-dirs",
            ",".join(run_dirs),
            "--output-csv",
            f"{prefix}_results_summary.csv",
            "--output-tex",
            f"{prefix}_results_table.tex",
        ]
    )

    print("\nFollow-up automation complete.")
    print(f"Results CSV: {prefix}_results_summary.csv")
    print(f"LaTeX table: {prefix}_results_table.tex")
    print(f"Transition analysis: baseline_runs/{args.region_name}_transition_analysis")


if __name__ == "__main__":
    main()
