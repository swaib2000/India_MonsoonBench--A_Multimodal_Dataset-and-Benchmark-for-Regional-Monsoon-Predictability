#!/usr/bin/env python3
"""
Collect model metrics into CSV and LaTeX table rows.

The script expects each run directory to contain metrics.json. If available, it
also reads analysis_summary.json and extreme_error_analysis/summary.json to add
region/state and extreme-regime evidence for rebuttal reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect rebuttal model results")
    parser.add_argument("--run-dirs", required=True, help="Comma-separated run directories")
    parser.add_argument("--output-csv", default="baseline_runs/northwest_monthly_ar_results_summary.csv")
    parser.add_argument("--output-tex", default="baseline_runs/northwest_monthly_ar_results_table.tex")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def fmt(value, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def run_label(run_dir: Path, metrics: dict) -> str:
    model = metrics.get("model", run_dir.name)
    loss = metrics.get("loss", "")
    if "optuna" in str(run_dir):
        return f"{model} ({loss}, tuned)"
    return f"{model} ({loss})"


def collect_row(run_dir_text: str) -> dict[str, object]:
    run_dir = Path(run_dir_text)
    metrics = load_json(run_dir / "metrics.json")
    analysis = load_json(run_dir / "analysis_summary.json")
    extreme = load_json(run_dir / "extreme_error_analysis" / "summary.json")
    ordinal = load_json(run_dir / "ordinal_metrics.json")

    return {
        "run_dir": str(run_dir),
        "label": run_label(run_dir, metrics),
        "model": metrics.get("model", ""),
        "loss": metrics.get("loss", ""),
        "best_epoch": metrics.get("best_epoch", ""),
        "best_val_acc": metrics.get("best_val_acc", ""),
        "test_acc": metrics.get("test_acc", ""),
        "test_weighted_f1": metrics.get("test_weighted_f1", ""),
        "test_loss": metrics.get("test_loss", ""),
        "num_train": metrics.get("num_train", ""),
        "num_val": metrics.get("num_val", ""),
        "num_test": metrics.get("num_test", ""),
        "scarcity_recall": extreme.get("scarcity_recall", ""),
        "large_excess_recall": extreme.get("large_excess_recall", ""),
        "within_1_accuracy": ordinal.get("within_1_accuracy", ""),
        "mean_abs_class_error": ordinal.get("mean_abs_class_error", ""),
        "severe_error_rate_ge2": ordinal.get("severe_error_rate_ge2", ""),
        "opposite_extreme_rate": ordinal.get("opposite_extreme_rate", ""),
        "scarcity_lowest_recall_region": extreme.get("scarcity_lowest_recall_region", ""),
        "large_excess_lowest_recall_region": extreme.get("large_excess_lowest_recall_region", ""),
        "analysis_num_predictions": analysis.get("num_predictions", ""),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_tex(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Northwest-Himalayan monthly autoregressive forecasting results. Inputs are three previous months of six dynamic modalities plus static elevation; target is next-month rainfall class.}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Val Acc. & Test Acc. & Test W-F1 & Within-1 & Severe $\geq$2 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(str(row['label']))} & "
            f"{fmt(row['best_val_acc'])} & "
            f"{fmt(row['test_acc'])} & "
            f"{fmt(row['test_weighted_f1'])} & "
            f"{fmt(row['within_1_accuracy'])} & "
            f"{fmt(row['severe_error_rate_ge2'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    run_dirs = [token.strip() for token in args.run_dirs.split(",") if token.strip()]
    if not run_dirs:
        raise ValueError("--run-dirs did not contain any paths")
    rows = [collect_row(run_dir) for run_dir in run_dirs]
    rows.sort(key=lambda row: float(row["best_val_acc"] or -1), reverse=True)
    write_csv(Path(args.output_csv), rows)
    write_tex(Path(args.output_tex), rows)
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_tex}")


if __name__ == "__main__":
    main()
