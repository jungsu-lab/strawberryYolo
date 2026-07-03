#!/usr/bin/env python3
"""Evaluate a trained YOLO strawberry detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_ultralytics_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Ultralytics is required for actual YOLO evaluation. "
            "Install local dependencies with: pip install -r requirements.txt"
        ) from exc
    return YOLO


def build_val_config(
    data_yaml: Path,
    model_path: Path,
    imgsz: int = 1280,
    batch: int = 8,
    device: str | int | None = None,
    project: Path = Path("runs/yolo26m"),
    name: str = "val",
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": str(model_path),
        "data": str(data_yaml),
        "imgsz": imgsz,
        "batch": batch,
        "project": str(project),
        "name": name,
    }
    if device is not None and str(device) != "":
        config["device"] = device
    return config


def eval_yolo26m_strawberry(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "val_config": config}

    YOLO = load_ultralytics_yolo()
    model = YOLO(str(config["model"]))
    val_kwargs = {key: value for key, value in config.items() if key != "model"}
    result = model.val(**val_kwargs)
    return {"dry_run": False, "result": result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO strawberry detector.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=Path("runs/yolo26m"))
    parser.add_argument("--name", default="val")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_val_config(
        data_yaml=args.data,
        model_path=args.model,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    result = eval_yolo26m_strawberry(config, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
