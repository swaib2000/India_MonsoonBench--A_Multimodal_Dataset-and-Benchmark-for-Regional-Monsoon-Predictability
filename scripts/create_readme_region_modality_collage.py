#!/usr/bin/env python3
"""
Create region-specific README figures from existing regional modality mosaics.

This script crops the already-rendered regional figures into individual
modality panels and then assembles a rectangular 4-region x 7-modality collage
for the GitHub README. It intentionally does not require geopandas because the
source regional mosaics already contain the corrected state boundaries/masks.

Example:

python scripts/create_readme_region_modality_collage.py \
  --source-dir website/assets/images/regions \
  --output-dir website/assets/images/readme
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REGIONS = [
    ("northwest_himalayan", "Northwest-Himalayan"),
    ("central_monsoon_core", "Central Monsoon Core"),
    ("south_peninsular_deccan", "South Peninsular-Deccan"),
    ("east_northeast_humid_orographic", "East-Northeast Orographic"),
]

MODALITIES = [
    ("elevation", "Elevation", 0, 0),
    ("rainfall_class", "Rainfall Class", 0, 1),
    ("lst", "LST", 0, 2),
    ("ndvi", "NDVI", 0, 3),
    ("rh", "Humidity", 1, 0),
    ("soil_moisture", "Soil Moisture", 1, 1),
    ("wind_speed", "Wind Speed", 1, 2),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop region-specific modality images for README")
    parser.add_argument("--source-dir", default="website/assets/images/regions")
    parser.add_argument("--output-dir", default="website/assets/images/readme")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month-name", default="sep")
    parser.add_argument("--panel-width", type=int, default=430)
    parser.add_argument("--panel-height", type=int, default=265)
    parser.add_argument("--title-height", type=int, default=120)
    parser.add_argument("--row-label-width", type=int, default=170)
    parser.add_argument("--padding", type=int, default=22)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def region_source_path(source_dir: Path, region: str, year: int, month_name: str) -> Path:
    return source_dir / f"{region}_{year}_{month_name}_modalities.png"


def crop_panel(source: Image.Image, row: int, col: int) -> Image.Image:
    """Crop a panel from the 2 x 4 regional mosaic layout."""
    width, height = source.size

    # The source plots use a suptitle and a small footer note. These fractions
    # keep panel titles/colorbars while removing most page-level whitespace.
    top = int(height * 0.065)
    bottom = int(height * 0.975)
    usable_height = bottom - top
    row_height = usable_height / 2.0
    col_width = width / 4.0

    pad_x = int(width * 0.004)
    pad_y = int(height * 0.006)
    left = int(col * col_width) + pad_x
    right = int((col + 1) * col_width) - pad_x
    upper = int(top + row * row_height) + pad_y
    lower = int(top + (row + 1) * row_height) - pad_y
    return source.crop((left, upper, right, lower))


def save_individual_panels(
    source_dir: Path,
    output_dir: Path,
    year: int,
    month_name: str,
) -> dict[tuple[str, str], Image.Image]:
    panel_images: dict[tuple[str, str], Image.Image] = {}
    panel_root = output_dir / "region_modalities"
    panel_root.mkdir(parents=True, exist_ok=True)

    for region, _label in REGIONS:
        source_path = region_source_path(source_dir, region, year, month_name)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source = Image.open(source_path).convert("RGB")
        region_dir = panel_root / region
        region_dir.mkdir(parents=True, exist_ok=True)

        for key, _title, row, col in MODALITIES:
            panel = crop_panel(source, row, col)
            panel_images[(region, key)] = panel
            path = region_dir / f"{key}.png"
            panel.save(path, optimize=True)
            print(f"Wrote {path}")

    return panel_images


def center_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, font, fill: str) -> None:
    left, top, right, bottom = xy
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.multiline_text(
        (left + (right - left - tw) / 2, top + (bottom - top - th) / 2),
        text,
        font=font,
        fill=fill,
        spacing=4,
        align="center",
    )


def save_collage(
    panel_images: dict[tuple[str, str], Image.Image],
    output_dir: Path,
    panel_width: int,
    panel_height: int,
    title_height: int,
    row_label_width: int,
    padding: int,
) -> Path:
    cols = len(MODALITIES)
    rows = len(REGIONS)
    header_height = 54
    width = row_label_width + cols * panel_width + (cols + 1) * padding
    height = title_height + header_height + rows * panel_height + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), "#f8faf7")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(36, bold=True)
    subtitle_font = load_font(18)
    header_font = load_font(16, bold=True)
    row_font = load_font(17, bold=True)

    center_text(
        draw,
        (0, 16, width, 62),
        "India MonsoonBench Regional Modalities",
        title_font,
        "#17202a",
    )
    center_text(
        draw,
        (0, 68, width, title_height - 10),
        "Four hydroclimatic regimes with state-bounded multimodal predictor mosaics",
        subtitle_font,
        "#455a64",
    )

    x0 = row_label_width + padding
    y_header = title_height
    for col_idx, (_key, title, _row, _col) in enumerate(MODALITIES):
        left = x0 + col_idx * (panel_width + padding)
        center_text(draw, (left, y_header, left + panel_width, y_header + header_height), title, header_font, "#263238")

    y0 = title_height + header_height + padding
    for row_idx, (region, label) in enumerate(REGIONS):
        row_top = y0 + row_idx * (panel_height + padding)
        center_text(
            draw,
            (padding, row_top, row_label_width, row_top + panel_height),
            label.replace(" ", "\n", 1),
            row_font,
            "#263238",
        )
        for col_idx, (key, _title, _row, _col) in enumerate(MODALITIES):
            panel = panel_images[(region, key)]
            panel = panel.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
            left = x0 + col_idx * (panel_width + padding)
            canvas.paste(panel, (left, row_top))

    output = output_dir / "monsoonbench_regional_readme_collage.png"
    canvas.save(output, optimize=True)
    print(f"Wrote {output}")
    return output


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_images = save_individual_panels(source_dir, output_dir, args.year, args.month_name)
    save_collage(
        panel_images,
        output_dir,
        args.panel_width,
        args.panel_height,
        args.title_height,
        args.row_label_width,
        args.padding,
    )


if __name__ == "__main__":
    main()
