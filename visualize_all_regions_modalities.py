#!/usr/bin/env python3
"""
Generate multimodal raster mosaic figures for all four climate regions.

This is a convenience wrapper around visualize_region_modalities.py. It creates
one figure per region for a selected year/month using:
  - elevation
  - rainfall class
  - LST
  - NDVI
  - relative humidity
  - soil moisture
  - wind speed

LULC is intentionally not included.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REGIONS = [
    "northwest_himalayan",
    "central_monsoon_core",
    "south_peninsular_deccan",
    "east_northeast_humid_orographic",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize all four regional modality mosaics")
    parser.add_argument("--manifest", default="modeling_manifest.csv")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=9)
    parser.add_argument("--output-dir", default="region_modality_maps_all")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--percentile-clip", default="2,98")
    parser.add_argument("--show-state-labels", action="store_true")
    parser.add_argument("--show-boundaries", action="store_true")
    parser.add_argument("--boundary-linewidth", type=float, default=0.8)
    parser.add_argument("--boundary-color", default="black")
    parser.add_argument("--fill-visual-nodata", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop if a region fails. Default continues to the next region.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successes: list[str] = []
    failures: dict[str, str] = {}

    for region in REGIONS:
        cmd = [
            sys.executable,
            "visualize_region_modalities.py",
            "--manifest",
            args.manifest,
            "--region",
            region,
            "--year",
            str(args.year),
            "--month",
            str(args.month),
            "--output-dir",
            str(output_dir),
            "--dpi",
            str(args.dpi),
            "--percentile-clip",
            args.percentile_clip,
            "--skip-missing",
        ]
        if args.show_state_labels:
            cmd.append("--show-state-labels")
        if args.show_boundaries:
            cmd.extend(
                [
                    "--show-boundaries",
                    "--boundary-linewidth",
                    str(args.boundary_linewidth),
                    "--boundary-color",
                    args.boundary_color,
                ]
            )
        if args.fill_visual_nodata:
            cmd.append("--fill-visual-nodata")

        print("\n$", " ".join(cmd), flush=True)
        completed = subprocess.run(cmd, check=False)
        if completed.returncode == 0:
            successes.append(region)
        else:
            failures[region] = f"exit_code={completed.returncode}"
            if args.fail_fast:
                raise SystemExit(completed.returncode)

    summary_path = output_dir / f"all_regions_{args.year}_{args.month:02d}_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"Year: {args.year}\n")
        f.write(f"Month: {args.month}\n")
        f.write(f"Manifest: {args.manifest}\n")
        f.write("\nSuccessful regions:\n")
        for region in successes:
            f.write(f"  - {region}\n")
        f.write("\nFailed regions:\n")
        for region, reason in failures.items():
            f.write(f"  - {region}: {reason}\n")

    print(f"\nWrote summary: {summary_path}")
    if failures:
        print("Some regions failed; see summary above.")
    else:
        print("All regional modality maps generated successfully.")


if __name__ == "__main__":
    main()
