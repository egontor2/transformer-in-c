#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np


def read_pgm(path):
    data = path.read_bytes()
    lines = data.split(b"\n")
    width, height = (int(value) for value in lines[1].split())
    offset = sum(len(line) + 1 for line in lines[:3])
    return np.frombuffer(data[offset:], dtype=np.uint8)[:width * height].reshape(
        height, width
    ).astype(np.float32) / 255.0


def load_split(root, split, classes):
    images = []
    labels = []
    with (root / f"{split}.csv").open(newline="") as manifest:
        for image_path, label_text in csv.reader(manifest):
            label = int(label_text)
            if label in classes:
                images.append(read_pgm(Path(image_path)))
                labels.append(label)
    return np.asarray(images), np.asarray(labels)


def evaluate_pair(images, labels, left, right):
    left_images = images[labels == left]
    right_images = images[labels == right]
    left_mean = left_images.mean(axis=0).reshape(-1)
    right_mean = right_images.mean(axis=0).reshape(-1)
    pair_images = images[(labels == left) | (labels == right)].reshape(-1, 784)
    pair_labels = labels[(labels == left) | (labels == right)]
    left_distance = ((pair_images - left_mean) ** 2).sum(axis=1)
    right_distance = ((pair_images - right_mean) ** 2).sum(axis=1)
    predictions = np.where(left_distance < right_distance, left, right)
    column_delta = left_images.mean(axis=(0, 1)) - right_images.mean(axis=(0, 1))
    return (
        float(np.mean(predictions == pair_labels)),
        float(np.max(np.abs(column_delta))),
        float(np.mean(np.abs(column_delta))),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/organ_smnist"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    for left, right in ((1, 2), (4, 5)):
        images, labels = load_split(args.root, args.split, {left, right})
        accuracy, maximum_delta, mean_delta = evaluate_pair(
            images, labels, left, right
        )
        print(
            f"{args.split} pair={left}/{right} "
            f"mean_distance_accuracy={accuracy:.4f} "
            f"max_column_delta={maximum_delta:.4f} "
            f"mean_abs_column_delta={mean_delta:.4f}"
        )


if __name__ == "__main__":
    main()
