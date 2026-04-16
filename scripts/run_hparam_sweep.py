#!/usr/bin/env python3
"""
Run a resumable hyperparameter sweep for patch-based rainfall benchmarks.

The script launches scripts/train_patch_baselines.py repeatedly, one configuration at a
time, then collects metrics.json from completed runs into a ranked summary.

Example:
    python scripts/run_hparam_sweep.py \
      --dataset-dir patch_dataset_monthly_ar_northwest_himalayan_full \
      --sweep-name northwest_himalayan_monthly_ar \
      --preset regional_ar_core \
      --epochs 30 \
      --batch-size 16 \
      --disable-early-stopping
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class TrialConfig:
    model: str
    loss: str
    lr: float
    weight_decay: float
    batch_size: int
    ordinal_weight: float = 0.5
    focal_gamma: float = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for scripts/train_patch_baselines.py")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--sweep-name", required=True)
    parser.add_argument("--output-root", default="baseline_runs/hparam_sweeps")
    parser.add_argument(
        "--preset",
        default="regional_ar_core",
        choices=["regional_ar_core", "conv3d_fast", "sota_small"],
        help="Predefined search space.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16, help="Fallback batch size for presets")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stats-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Pass --disable-early-stopping to every trial.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Optional limit for debugging. 0 runs the full preset.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Rerun trials even when metrics.json already exists.",
    )
    parser.add_argument(
        "--rank-by",
        default="test_weighted_f1",
        choices=["test_weighted_f1", "test_acc", "best_val_acc"],
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Pass --no-class-weights to every trial.",
    )
    return parser.parse_args()


def safe_float_name(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def trial_name(config: TrialConfig) -> str:
    parts = [
        config.model,
        config.loss,
        f"lr{safe_float_name(config.lr)}",
        f"wd{safe_float_name(config.weight_decay)}",
        f"bs{config.batch_size}",
    ]
    if "ordinal" in config.loss:
        parts.append(f"ow{safe_float_name(config.ordinal_weight)}")
    if "focal" in config.loss:
        parts.append(f"fg{safe_float_name(config.focal_gamma)}")
    return "_".join(parts)


def build_trials(preset: str, fallback_batch_size: int) -> list[TrialConfig]:
    if preset == "conv3d_fast":
        return [
            TrialConfig(model="conv3d", loss=loss, lr=lr, weight_decay=wd, batch_size=fallback_batch_size, ordinal_weight=ow)
            for loss, lr, wd, ow in itertools.product(
                ["ce", "focal", "ce_ordinal"],
                [1e-3, 3e-4],
                [1e-4, 1e-3],
                [0.3],
            )
        ]

    if preset == "sota_small":
        trials: list[TrialConfig] = []
        for model in ["conv3d", "convlstm", "swin3d"]:
            batch = 8 if model == "swin3d" else fallback_batch_size
            for loss in ["ce", "focal", "ce_ordinal"]:
                for lr in [3e-4, 1e-4]:
                    trials.append(
                        TrialConfig(
                            model=model,
                            loss=loss,
                            lr=lr,
                            weight_decay=1e-3,
                            batch_size=batch,
                            ordinal_weight=0.3,
                        )
                    )
        return trials

    # regional_ar_core: balanced enough to improve results without exploding runtime.
    trials = []
    for model in ["conv3d", "convlstm"]:
        for loss in ["ce", "focal", "ce_ordinal"]:
            for lr in [1e-3, 3e-4, 1e-4]:
                for wd in [1e-4, 1e-3]:
                    trials.append(
                        TrialConfig(
                            model=model,
                            loss=loss,
                            lr=lr,
                            weight_decay=wd,
                            batch_size=fallback_batch_size,
                            ordinal_weight=0.3,
                        )
                    )

    # Add a few transformer trials, but keep them deliberately limited.
    for loss in ["ce", "focal", "ce_ordinal"]:
        for lr in [3e-4, 1e-4]:
            trials.append(
                TrialConfig(
                    model="swin3d",
                    loss=loss,
                    lr=lr,
                    weight_decay=1e-3,
                    batch_size=8,
                    ordinal_weight=0.3,
                )
            )
    return trials


def run_trial(
    config: TrialConfig,
    args: argparse.Namespace,
    output_dir: Path,
    log_path: Path,
) -> int:
    cmd = [
        sys.executable,
        "scripts/train_patch_baselines.py",
        "--dataset-dir",
        args.dataset_dir,
        "--output-dir",
        str(output_dir),
        "--model",
        config.model,
        "--loss",
        config.loss,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(config.batch_size),
        "--lr",
        str(config.lr),
        "--weight-decay",
        str(config.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--stats-samples",
        str(args.stats_samples),
        "--seed",
        str(args.seed),
        "--ordinal-weight",
        str(config.ordinal_weight),
        "--focal-gamma",
        str(config.focal_gamma),
    ]
    if args.disable_early_stopping:
        cmd.append("--disable-early-stopping")
    if args.no_class_weights:
        cmd.append("--no-class-weights")

    print("\n" + "=" * 96, flush=True)
    print("Running:", trial_name(config), flush=True)
    print("Command:", " ".join(cmd), flush=True)
    print("=" * 96, flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return process.wait()


def collect_rows(sweep_dir: Path, rank_by: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(sweep_dir.glob("*/metrics.json")):
        with metrics_path.open() as f:
            metrics = json.load(f)
        trial_dir = metrics_path.parent
        rows.append(
            {
                "trial": trial_dir.name,
                "run_dir": str(trial_dir),
                "model": metrics.get("model"),
                "loss": metrics.get("loss"),
                "ordinal_weight": metrics.get("ordinal_weight"),
                "focal_gamma": metrics.get("focal_gamma"),
                "best_epoch": metrics.get("best_epoch"),
                "best_val_acc": metrics.get("best_val_acc"),
                "test_acc": metrics.get("test_acc"),
                "test_weighted_f1": metrics.get("test_weighted_f1"),
                "num_train": metrics.get("num_train"),
                "num_val": metrics.get("num_val"),
                "num_test": metrics.get("num_test"),
                "status": "completed",
            }
        )
    return sorted(rows, key=lambda row: float(row.get(rank_by) or -1), reverse=True)


def write_summary(sweep_dir: Path, rows: list[dict[str, object]]) -> None:
    summary_csv = sweep_dir / "sweep_summary.csv"
    summary_json = sweep_dir / "sweep_summary.json"
    if rows:
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with summary_json.open("w") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.output_root) / args.sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    trials = build_trials(args.preset, args.batch_size)
    if args.max_trials > 0:
        trials = trials[: args.max_trials]

    with (sweep_dir / "sweep_config.json").open("w") as f:
        json.dump(
            {
                "args": vars(args),
                "num_trials": len(trials),
                "trials": [asdict(t) | {"name": trial_name(t)} for t in trials],
            },
            f,
            indent=2,
        )

    failures: list[dict[str, object]] = []
    for i, config in enumerate(trials, start=1):
        name = trial_name(config)
        output_dir = sweep_dir / name
        log_path = output_dir / "train.log"
        metrics_path = output_dir / "metrics.json"

        print(f"\nTrial {i}/{len(trials)}: {name}", flush=True)
        if metrics_path.exists() and not args.rerun_completed:
            print(f"Skipping completed trial: {metrics_path}", flush=True)
            continue

        returncode = run_trial(config, args, output_dir, log_path)
        if returncode != 0:
            print(f"Trial failed with code {returncode}: {name}", flush=True)
            failures.append({"trial": name, "returncode": returncode, "log": str(log_path)})

        rows = collect_rows(sweep_dir, args.rank_by)
        write_summary(sweep_dir, rows)
        if rows:
            best = rows[0]
            print(
                f"Current best by {args.rank_by}: {best['trial']} "
                f"val={best['best_val_acc']} test_acc={best['test_acc']} wf1={best['test_weighted_f1']}",
                flush=True,
            )

    rows = collect_rows(sweep_dir, args.rank_by)
    write_summary(sweep_dir, rows)
    with (sweep_dir / "failed_trials.json").open("w") as f:
        json.dump(failures, f, indent=2)

    print("\nSweep complete.")
    print(f"Summary CSV:  {sweep_dir / 'sweep_summary.csv'}")
    print(f"Summary JSON: {sweep_dir / 'sweep_summary.json'}")
    if failures:
        print(f"Failures: {len(failures)}. See {sweep_dir / 'failed_trials.json'}")
    if rows:
        best = rows[0]
        print(
            f"Best by {args.rank_by}: {best['trial']} | "
            f"val={best['best_val_acc']} test_acc={best['test_acc']} wf1={best['test_weighted_f1']}"
        )


if __name__ == "__main__":
    main()
