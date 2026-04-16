#!/usr/bin/env python3
"""
Optuna hyperparameter tuning wrapper for scripts/train_patch_baselines.py.

This script launches one training run per Optuna trial, reads the resulting
metrics.json, and optimizes a validation metric. It intentionally treats the
test metric as reporting-only so that hyperparameter selection remains based on
the validation year.

Example:
    python scripts/tune_patch_baselines_optuna.py \
      --dataset-dir patch_dataset_monthly_ar_northwest_himalayan_full \
      --study-name northwest_monthly_ar_conv3d \
      --output-root baseline_runs/optuna_northwest_monthly_ar \
      --models conv3d \
      --losses ce,focal,ce_ordinal \
      --n-trials 20 \
      --epochs 20 \
      --batch-sizes 16,32 \
      --disable-early-stopping
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune patch baselines with Optuna.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-root", default="baseline_runs/optuna_patch_tuning")
    parser.add_argument("--study-name", default="patch_baseline_tuning")
    parser.add_argument(
        "--storage",
        default="",
        help="Optuna storage URL. Default: sqlite:///<output-root>/<study-name>.db",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0, help="Optional tuning timeout in seconds.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--models", default="conv3d", help="Comma-separated model choices.")
    parser.add_argument("--losses", default="ce,focal,ce_ordinal", help="Comma-separated loss choices.")
    parser.add_argument("--batch-sizes", default="16,32", help="Comma-separated batch-size choices.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stats-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="best_val_acc", choices=["best_val_acc", "test_acc", "test_weighted_f1"])
    parser.add_argument(
        "--direction",
        default="maximize",
        choices=["maximize", "minimize"],
        help="Optimization direction. Usually maximize for accuracy/F1.",
    )
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=3e-3)
    parser.add_argument("--weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay-max", type=float, default=3e-3)
    parser.add_argument("--focal-gamma-min", type=float, default=1.0)
    parser.add_argument("--focal-gamma-max", type=float, default=4.0)
    parser.add_argument("--ordinal-weight-min", type=float, default=0.1)
    parser.add_argument("--ordinal-weight-max", type=float, default=0.8)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Run every trial for the full epoch count.",
    )
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one sampled command and exit without running Optuna.",
    )
    return parser.parse_args()


def split_csv(text: str) -> list[str]:
    values = [token.strip() for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError(f"No values parsed from: {text!r}")
    return values


def split_int_csv(text: str) -> list[int]:
    return [int(value) for value in split_csv(text)]


def require_optuna():
    try:
        import optuna  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Optuna is required. Install it in your training environment with:\n"
            "  pip install optuna\n"
            "or:\n"
            "  conda install -c conda-forge optuna"
        ) from exc
    return optuna


def command_to_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def build_trial_command(
    args: argparse.Namespace,
    trial,
    output_dir: Path,
) -> tuple[list[str], dict[str, object]]:
    models = split_csv(args.models)
    losses = split_csv(args.losses)
    batch_sizes = split_int_csv(args.batch_sizes)

    model = trial.suggest_categorical("model", models)
    loss = trial.suggest_categorical("loss", losses)
    batch_size = trial.suggest_categorical("batch_size", batch_sizes)
    lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
    weight_decay = trial.suggest_float(
        "weight_decay", args.weight_decay_min, args.weight_decay_max, log=True
    )

    focal_gamma = 2.0
    if "focal" in loss:
        focal_gamma = trial.suggest_float("focal_gamma", args.focal_gamma_min, args.focal_gamma_max)

    ordinal_weight = 0.5
    if "ordinal" in loss:
        ordinal_weight = trial.suggest_float(
            "ordinal_weight", args.ordinal_weight_min, args.ordinal_weight_max
        )

    cmd = [
        sys.executable,
        "scripts/train_patch_baselines.py",
        "--dataset-dir",
        args.dataset_dir,
        "--output-dir",
        str(output_dir),
        "--model",
        model,
        "--loss",
        loss,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--stats-samples",
        str(args.stats_samples),
        "--seed",
        str(args.seed + trial.number),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--focal-gamma",
        str(focal_gamma),
        "--ordinal-weight",
        str(ordinal_weight),
    ]

    if args.disable_early_stopping:
        cmd.append("--disable-early-stopping")
    if args.no_class_weights:
        cmd.append("--no-class-weights")

    params = {
        "model": model,
        "loss": loss,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "focal_gamma": focal_gamma,
        "ordinal_weight": ordinal_weight,
    }
    return cmd, params


def read_metrics(path: Path) -> dict[str, object]:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Training finished but metrics.json was not found: {metrics_path}")
    with metrics_path.open() as f:
        return json.load(f)


def write_leaderboard(output_root: Path, study) -> None:
    rows = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
        }
        row.update(trial.params)
        row.update(trial.user_attrs)
        rows.append(row)

    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row})
    path = output_root / "leaderboard.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    optuna = require_optuna()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{output_root / (args.study_name + '.db')}"

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=args.direction,
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial):
        trial_dir = output_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        cmd, params = build_trial_command(args, trial, trial_dir)

        if args.dry_run:
            print(command_to_string(cmd))
            raise SystemExit(0)

        command_path = trial_dir / "command.txt"
        command_path.write_text(command_to_string(cmd) + "\n")
        print(f"\nTrial {trial.number}: {params}")
        print(command_to_string(cmd), flush=True)

        log_path = trial_dir / "train.log"
        with log_path.open("w") as log_file:
            completed = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode != 0:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text().splitlines()[-30:])
            except Exception:
                pass
            raise RuntimeError(f"Training failed for trial {trial.number}. Log tail:\n{tail}")

        metrics = read_metrics(trial_dir)
        for key in [
            "best_epoch",
            "best_val_acc",
            "test_acc",
            "test_weighted_f1",
            "test_loss",
            "num_train",
            "num_val",
            "num_test",
        ]:
            if key in metrics:
                trial.set_user_attr(key, metrics[key])
        trial.set_user_attr("output_dir", str(trial_dir))
        trial.set_user_attr("command", command_to_string(cmd))

        write_leaderboard(output_root, study)
        return float(metrics[args.metric])

    timeout = args.timeout if args.timeout > 0 else None
    study.optimize(objective, n_trials=args.n_trials, timeout=timeout)
    write_leaderboard(output_root, study)

    print("\nBest trial")
    print(f"  number: {study.best_trial.number}")
    print(f"  value:  {study.best_value}")
    print(f"  params: {study.best_trial.params}")
    print(f"  attrs:  {study.best_trial.user_attrs}")
    print(f"\nLeaderboard: {output_root / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
