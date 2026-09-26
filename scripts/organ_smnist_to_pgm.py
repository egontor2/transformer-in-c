#!/usr/bin/env python3
"""Convert a MedMNIST OrganSMNIST NPZ split to PGM files and a CSV manifest."""

import argparse
import csv
from pathlib import Path

import numpy as np


def convert_split(npz_path: Path, split: str, output_dir: Path) -> None:
    archive = np.load(npz_path)
    image_key = f"{split}_images"
    label_key = f"{split}_labels"
    if image_key not in archive or label_key not in archive:
        raise ValueError(f"missing {image_key} or {label_key} in {npz_path}")

    images = archive[image_key]
    labels = archive[label_key].reshape(-1)
    if images.ndim != 3 or images.shape[0] != labels.shape[0]:
        raise ValueError("expected images [samples][height][width] and labels")
    if images.dtype.kind not in "uib":
        raise ValueError("expected integer grayscale images")

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{split}.csv"
    with manifest_path.open("w", newline="") as manifest:
        writer = csv.writer(manifest)
        for index, (image, label) in enumerate(zip(images, labels)):
            image_path = split_dir / f"{index:06d}.pgm"
            height, width = image.shape
            with image_path.open("wb") as pgm:
                pgm.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
                pgm.write(np.asarray(image, dtype=np.uint8).tobytes())
            writer.writerow((image_path, int(label)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        choices=("train", "val", "test"),
    )
    args = parser.parse_args()
    for split in args.splits:
        convert_split(args.npz, split, args.output)


if __name__ == "__main__":
    main()
