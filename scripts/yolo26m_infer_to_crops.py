#!/usr/bin/env python3
"""Run YOLO inference and export fixed-size object crops."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def load_ultralytics_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Ultralytics is required for actual YOLO inference. "
            "Install local dependencies with: pip install -r requirements.txt"
        ) from exc
    return YOLO


def crop_with_margin(image: Image.Image, xyxy: list[float], margin: float = 0.30, output_size: int = 256) -> Image.Image:
    """Crop a square region around an xyxy box, clamp to image bounds, and resize."""

    if len(xyxy) != 4:
        raise ValueError("xyxy must contain exactly four values")
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid xyxy box: {xyxy}")
    width = x2 - x1
    height = y2 - y1
    side = max(width, height) * (1.0 + margin)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    nx1 = max(0, int(round(cx - side / 2.0)))
    ny1 = max(0, int(round(cy - side / 2.0)))
    nx2 = min(image.width, int(round(cx + side / 2.0)))
    ny2 = min(image.height, int(round(cy + side / 2.0)))
    if nx2 <= nx1 or ny2 <= ny1:
        raise ValueError(f"crop outside image bounds for box: {xyxy}")
    return image.crop((nx1, ny1, nx2, ny2)).resize((output_size, output_size))


def export_crops_from_detection_rows(
    detections: pd.DataFrame,
    output_dir: Path,
    margin: float = 0.30,
    output_size: int = 256,
) -> pd.DataFrame:
    """Export crops from a detection table containing image_path and x1/y1/x2/y2 columns."""

    required = {"image_path", "x1", "y1", "x2", "y2"}
    missing = required - set(detections.columns)
    if missing:
        raise ValueError(f"detections missing required columns: {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, row in detections.reset_index(drop=True).iterrows():
        image_path = Path(str(row["image_path"]))
        with Image.open(image_path).convert("RGB") as image:
            crop = crop_with_margin(
                image,
                [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                margin=margin,
                output_size=output_size,
            )
        crop_path = output_dir / f"{image_path.stem}_{idx:05d}.jpg"
        crop.save(crop_path)
        output_row = row.to_dict()
        output_row["crop_path"] = str(crop_path)
        output_row["crop_size"] = output_size
        rows.append(output_row)
    return pd.DataFrame(rows)


def run_yolo_inference_to_crops(
    model_path: Path,
    source: Path,
    output_dir: Path,
    imgsz: int = 1280,
    conf: float = 0.20,
    margin: float = 0.30,
    output_size: int = 256,
) -> pd.DataFrame:
    """Run Ultralytics YOLO prediction and save crop images."""

    YOLO = load_ultralytics_yolo()
    model = YOLO(str(model_path))
    results = model.predict(str(source), imgsz=imgsz, conf=conf, verbose=False)
    rows: list[dict[str, Any]] = []
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        image_path = Path(result.path)
        with Image.open(image_path).convert("RGB") as image:
            for det_idx, box in enumerate(result.boxes):
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                crop = crop_with_margin(image, xyxy, margin=margin, output_size=output_size)
                crop_path = crop_dir / f"{image_path.stem}_{det_idx:05d}.jpg"
                crop.save(crop_path)
                rows.append(
                    {
                        "image_path": str(image_path),
                        "det_id": det_idx,
                        "class_id": int(box.cls.item()),
                        "conf": float(box.conf.item()),
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                        "crop_path": str(crop_path),
                    }
                )
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "detections_with_crops.csv", index=False, encoding="utf-8")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference and save object crops.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/yolocsv/yolo_crops"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--margin", type=float, default=0.30)
    parser.add_argument("--output-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = run_yolo_inference_to_crops(
        model_path=args.model,
        source=args.source,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        margin=args.margin,
        output_size=args.output_size,
    )
    print(f"wrote {len(frame)} crops to {args.output_dir}")


if __name__ == "__main__":
    main()
