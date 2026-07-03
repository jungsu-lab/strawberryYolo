#!/usr/bin/env python3
"""Draw YOLO labels on sample images for visual sanity checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class SanityReportSummary:
    image_count: int
    box_count: int
    output_dir: Path


COLORS = [
    (220, 38, 38),
    (37, 99, 235),
    (22, 163, 74),
    (217, 119, 6),
    (147, 51, 234),
    (14, 116, 144),
    (190, 18, 60),
    (82, 82, 91),
]


def _load_dataset_yaml(dataset_yaml: Path) -> dict[str, Any]:
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid dataset yaml: {dataset_yaml}")
    return data


def _dataset_root(dataset_yaml: Path, data: dict[str, Any]) -> Path:
    root = Path(str(data.get("path", "")))
    if not root:
        return dataset_yaml.parent
    if not root.is_absolute():
        return (dataset_yaml.parent / root).resolve()
    return root


def _names(data: dict[str, Any]) -> dict[int, str]:
    raw_names = data.get("names", {})
    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    return {}


def _read_label_file(label_path: Path, width: int, height: int) -> list[dict[str, float | int]]:
    boxes: list[dict[str, float | int]] = []
    if not label_path.exists():
        return boxes
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, box_w, box_h = (float(value) for value in parts[1:])
        except ValueError:
            continue
        x1 = (x_center - box_w / 2) * width
        y1 = (y_center - box_h / 2) * height
        x2 = (x_center + box_w / 2) * width
        y2 = (y_center + box_h / 2) * height
        boxes.append({"class_id": class_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "line_number": line_number})
    return boxes


def _draw_boxes(image_path: Path, label_path: Path, output_path: Path, class_names: dict[int, str]) -> int:
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        boxes = _read_label_file(label_path, image.width, image.height)
        for box in boxes:
            class_id = int(box["class_id"])
            color = COLORS[class_id % len(COLORS)]
            xy = [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
            draw.rectangle(xy, outline=color, width=3)
            label = f"{class_id}:{class_names.get(class_id, str(class_id))}"
            text_bbox = draw.textbbox((xy[0], xy[1]), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((xy[0], xy[1]), label, fill=(255, 255, 255), font=font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    return len(boxes)


def _image_files(image_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def build_yolo_label_sanity_report(
    dataset_yaml: Path,
    output_dir: Path,
    splits: list[str] | None = None,
    sample_count: int = 20,
) -> SanityReportSummary:
    data = _load_dataset_yaml(dataset_yaml)
    root = _dataset_root(dataset_yaml, data)
    class_names = _names(data)
    splits = splits or ["train", "val", "test"]

    rows: list[dict[str, Any]] = []
    total_images = 0
    total_boxes = 0
    for split in splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        for image_path in _image_files(image_dir)[:sample_count]:
            label_path = label_dir / f"{image_path.stem}.txt"
            output_path = output_dir / split / image_path.name
            box_count = _draw_boxes(image_path, label_path, output_path, class_names)
            rows.append({"split": split, "image": image_path.name, "label": label_path.name, "box_count": box_count, "output": str(output_path)})
            total_images += 1
            total_boxes += box_count

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["split", "image", "label", "box_count", "output"]).to_csv(
        output_dir / "label_sanity_summary.csv",
        index=False,
        encoding="utf-8",
    )
    return SanityReportSummary(image_count=total_images, box_count=total_boxes, output_dir=output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO labels on sample images.")
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/yolocsv/label_sanity_examples"))
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated split names.")
    parser.add_argument("--sample-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_yolo_label_sanity_report(
        dataset_yaml=args.dataset_yaml,
        output_dir=args.output_dir,
        splits=[split.strip() for split in args.splits.split(",") if split.strip()],
        sample_count=args.sample_count,
    )
    print(f"wrote {summary.image_count} sanity images with {summary.box_count} boxes to {summary.output_dir}")


if __name__ == "__main__":
    main()
