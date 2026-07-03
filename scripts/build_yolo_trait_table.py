#!/usr/bin/env python3
"""Build a trait-ready object table from YOLO/AIHub conversion metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _merge_image_features(annotation_frame: pd.DataFrame, image_features: pd.DataFrame) -> pd.DataFrame:
    if image_features.empty:
        return annotation_frame.copy()
    merge_columns = ["image_id"]
    feature_columns = [
        column
        for column in image_features.columns
        if column == "image_id" or column not in annotation_frame.columns
    ]
    return annotation_frame.merge(image_features[feature_columns], on=merge_columns, how="left")


def build_yolo_trait_table(
    annotation_manifest_csv: Path,
    image_growth_features_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Create object-level features used by trait/growth/yield stages."""

    annotations = _read_csv(annotation_manifest_csv)
    image_features = _read_csv(image_growth_features_csv)
    frame = _merge_image_features(annotations, image_features)

    required = {"crops_id", "image_id", "annotation_id", "bbox_w", "bbox_h", "image_width", "image_height"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"annotation manifest missing required columns: {sorted(missing)}")

    frame["bbox_w"] = _to_numeric(frame["bbox_w"])
    frame["bbox_h"] = _to_numeric(frame["bbox_h"])
    frame["image_width"] = _to_numeric(frame["image_width"])
    frame["image_height"] = _to_numeric(frame["image_height"])
    frame["bbox_area_px"] = frame["bbox_w"] * frame["bbox_h"]
    image_area = frame["image_width"] * frame["image_height"]
    frame["bbox_area_ratio"] = (frame["bbox_area_px"] / image_area).fillna(0.0)
    frame["bbox_aspect_ratio"] = (frame["bbox_w"] / frame["bbox_h"]).replace([float("inf"), -float("inf")], pd.NA).fillna(0.0)
    frame["fruit_id"] = (
        frame["crops_id"].astype(str)
        + ":"
        + frame["image_id"].astype(str)
        + ":"
        + frame["annotation_id"].astype(str)
    )

    preferred = [
        "fruit_id",
        "image_id",
        "fname",
        "split",
        "farm_id",
        "crops_id",
        "growth_stage",
        "date_captured",
        "annotation_id",
        "category_name",
        "class_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "image_width",
        "image_height",
        "bbox_area_px",
        "bbox_area_ratio",
        "bbox_aspect_ratio",
    ]
    columns = [column for column in preferred if column in frame.columns] + [
        column for column in frame.columns if column not in preferred
    ]
    frame = frame[columns]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, encoding="utf-8")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build object-level trait table from converted YOLO metadata.")
    parser.add_argument("--annotation-manifest", type=Path, required=True)
    parser.add_argument("--image-growth-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("processed/yolo/trait_table.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_yolo_trait_table(args.annotation_manifest, args.image_growth_features, args.output)
    print(f"wrote {len(frame)} trait rows to {args.output}")


if __name__ == "__main__":
    main()
