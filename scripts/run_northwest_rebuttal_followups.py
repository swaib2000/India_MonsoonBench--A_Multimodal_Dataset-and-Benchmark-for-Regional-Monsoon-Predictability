#!/usr/bin/env python3
"""
Run the next northwest-Himalayan rebuttal experiments after Optuna finishes.

Workflow:
  1. Wait until the Conv3D Optuna search has N complete trials.
  2. Select the best complete trial by validation accuracy.
  3. Run per-state/per-region and extreme-regime analyses for the best trial.
  4. Train ConvLSTM and Swin3D with the best tuned loss/LR/regularization.
  5. Analyze the new runs.
  6. Collect all metrics into CSV and LaTeX table snippets.

This script should be launched from the repo root inside the same conda
environment used for training.
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
    parser = argparse.ArgumentParser(description="Run northwest-Himalayan follow-up experiments")
    parser.add_argument("--dataset-dir", default="patch_dataset_monthly_ar_northwest_himalayan_full")
    parser.add_argument("--optuna-root", default="baseline_runs/optuna_northwest_monthly_ar")
    parser.add_argument("--expected-trials", type=int, default=24)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stats-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument(
        "--models",
        default="convlstm,swin3d",
        help="Comma-separated follow-up architectures to train after Optuna.",
    )
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
        rows = read_leaderboard(leaderboard)
        complete = complete_trials(rows)
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


def train_followup_model(
    args: argparse.Namespace,
    model: str,
    best: dict[str, str],
    run_dir: Path,
    log_name: str,
) -> None:
    batch_size = get_param(best, "batch_size", "16")
    if model in {"convlstm", "swin3d"} and int(float(batch_size)) > 16:
        batch_size = "16"

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
    run(cmd, Path(log_name))


def analyze_run(dataset_dir: str, run_dir: Path) -> None:
    run([sys.executable, "scripts/analyze_patch_predictions.py", "--dataset-dir", dataset_dir, "--run-dir", str(run_dir)])
    run([sys.executable, "scripts/analyze_extreme_regime_errors.py", "--dataset-dir", dataset_dir, "--run-dir", str(run_dir)])


def main() -> None:
    args = parse_args()
    optuna_root = Path(args.optuna_root)
    leaderboard = optuna_root / "leaderboard.csv"

    if args.skip_wait:
        complete = complete_trials(read_leaderboard(leaderboard))
    else:
        complete = wait_for_optuna(leaderboard, args.expected_trials, args.poll_seconds)

    best = best_trial(complete)
    best_run_dir = Path(best["output_dir"])
    summary = {
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
        run_dir = Path("baseline_runs") / f"{model}_monthly_ar_northwest_himalayan_tuned"
        log_name = f"train_{model}_monthly_ar_northwest_himalayan_tuned.log"
        train_followup_model(args, model, best, run_dir, log_name)
        analyze_run(args.dataset_dir, run_dir)
        run_dirs.append(str(run_dir))

    run(
        [
            sys.executable,
            "scripts/collect_rebuttal_model_results.py",
            "--run-dirs",
            ",".join(run_dirs),
            "--output-csv",
            "baseline_runs/northwest_monthly_ar_results_summary.csv",
            "--output-tex",
            "baseline_runs/northwest_monthly_ar_results_table.tex",
        ]
    )

    print("\nFollow-up automation complete.")
    print("Results CSV: baseline_runs/northwest_monthly_ar_results_summary.csv")
    print("LaTeX table: baseline_runs/northwest_monthly_ar_results_table.tex")


if __name__ == "__main__":
    main()
