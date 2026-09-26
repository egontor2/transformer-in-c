#!/usr/bin/env python3
"""Convert grayscale image folders grouped by class into PGM manifests."""

import argparse
import csv
from pathlib import Path

from PIL import Image


EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def convert_split(input_dir: Path, output_dir: Path, split: str) -> None:
    source = input_dir / split
    if not source.is_dir():
        raise ValueError(f"missing split directory: {source}")
    destination = output_dir / split
    destination.mkdir(parents=True, exist_ok=True)
    class_dirs = sorted(
        (path for path in source.iterdir() if path.is_dir()),
        key=lambda path: (
            (0, int(path.name)) if path.name.isdigit() else (1, path.name)
        ),
    )
    if not class_dirs:
        raise ValueError(f"expected class directories in {source}")
    manifest_path = output_dir / f"{split}.csv"
    index = 0
    with manifest_path.open("w", newline="") as manifest:
        writer = csv.writer(manifest)
        for label, class_dir in enumerate(class_dirs):
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in EXTENSIONS:
                    continue
                with Image.open(image_path) as image:
                    grayscale = image.convert("L")
                    if grayscale.size != (28, 28):
                        grayscale = grayscale.resize((28, 28), Image.Resampling.BILINEAR)
                    output_path = destination / f"{index:06d}.pgm"
                    grayscale.save(output_path, format="PPM")
                writer.writerow((output_path, label))
                index += 1
    if index == 0:
        raise ValueError(f"no supported images found in {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--splits", nargs="+", default=("train", "val", "test"),
        choices=("train", "val", "test"),
    )
    args = parser.parse_args()
    for split in args.splits:
        convert_split(args.input, args.output, split)


if __name__ == "__main__":
    main()
