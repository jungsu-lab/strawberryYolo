#!/usr/bin/env python3
"""Build a reproducible YOLO ablation command plan."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd


def _model_stem(model: str) -> str:
    return Path(model).stem


def _conf_label(conf: float) -> str:
    return f"{int(round(conf * 100)):03d}"


def build_yolo_ablation_plan(
    data_yaml: Path,
    output_csv: Path,
    models: list[str],
    image_sizes: list[int],
    conf_thresholds: list[float],
    epochs: int = 120,
    batch: int = -1,
    device: str = "0",
    project: str = "runs/yolo_ablation",
) -> pd.DataFrame:
    """Write train/eval/crop commands for detector ablation experiments."""

    rows: list[dict[str, str | int | float]] = []
    for model, imgsz, conf in product(models, image_sizes, conf_thresholds):
        model_name = _model_stem(model)
        experiment_id = f"{model_name}_img{imgsz}_conf{_conf_label(conf)}"
        run_dir = f"{project}/{experiment_id}"
        train_command = (
            "python scripts/train_yolo26m_strawberry.py "
            f"--data {data_yaml} "
            f"--model {model} "
            f"--imgsz {imgsz} "
            f"--epochs {epochs} "
            f"--batch {batch} "
            f"--device {device} "
            f"--project {project} "
            f"--name {experiment_id}"
        )
        eval_command = (
            "python scripts/eval_yolo26m_strawberry.py "
            f"--data {data_yaml} "
            f"--model {run_dir}/weights/best.pt "
            f"--imgsz {imgsz} "
            f"--batch 8 "
            f"--device {device} "
            f"--project {project} "
            f"--name {experiment_id}_val"
        )
        crop_command = (
            "python scripts/yolo26m_infer_to_crops.py "
            f"--model {run_dir}/weights/best.pt "
            "--source /path/to/images_or_video "
            f"--output-dir reports/yolocsv/yolo_crops/{experiment_id} "
            f"--imgsz {imgsz} "
            f"--conf {conf} "
            "--margin 0.30 "
            "--output-size 256"
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "model": model,
                "imgsz": imgsz,
                "conf": conf,
                "train_command": train_command,
                "eval_command": eval_command,
                "crop_command": crop_command,
            }
        )
    frame = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, encoding="utf-8")
    return frame


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO ablation train/eval/crop command plan.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/yolocsv/yolo_ablation_plan.csv"))
    parser.add_argument("--models", type=_parse_csv_list, default=["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"])
    parser.add_argument("--imgsz", type=_parse_int_list, default=[640, 960, 1280])
    parser.add_argument("--conf", type=_parse_float_list, default=[0.10, 0.20, 0.30])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/yolo_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_yolo_ablation_plan(
        data_yaml=args.data,
        output_csv=args.output,
        models=args.models,
        image_sizes=args.imgsz,
        conf_thresholds=args.conf,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
    )
    print(f"wrote {len(frame)} ablation commands to {args.output}")


if __name__ == "__main__":
    main()
