#!/usr/bin/env python3
"""Create a lightweight yield baseline from YOLO/KGCV sequence features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FRUIT_LABELS = {"과실", "fruit", "green", "white", "turning_red", "red", "overripe", "small_g"}


def _numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _is_fruit(frame: pd.DataFrame) -> pd.Series:
    if "category_name" in frame.columns:
        return frame["category_name"].astype(str).isin(FRUIT_LABELS)
    if "class_name" in frame.columns:
        return frame["class_name"].astype(str).isin(FRUIT_LABELS)
    if "stage_score" in frame.columns:
        return _numeric(frame["stage_score"]) >= 3.0
    return pd.Series([True] * len(frame), index=frame.index)


def _predicted_weight(frame: pd.DataFrame) -> pd.Series:
    if "growth_fr_weight" in frame.columns:
        measured = pd.to_numeric(frame["growth_fr_weight"], errors="coerce")
    else:
        measured = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Float64")
    if {"diameter_mm", "length_mm"}.issubset(frame.columns):
        diameter = _numeric(frame["diameter_mm"])
        length = _numeric(frame["length_mm"])
        proxy = 0.00052 * diameter * diameter * length
    elif {"bbox_w", "bbox_h"}.issubset(frame.columns):
        proxy = 0.01 * _numeric(frame["bbox_w"]) * _numeric(frame["bbox_h"])
    elif "bbox_area_ratio" in frame.columns:
        proxy = 100.0 * _numeric(frame["bbox_area_ratio"])
    else:
        proxy = pd.Series([0.0] * len(frame), index=frame.index)
    return measured.fillna(proxy).astype(float)


def predict_yolo_kgcv_yield(
    sequence_csv: Path,
    fruit_output_csv: Path,
    group_output_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict fruit-level and crop-level yield with a simple reproducible baseline."""

    frame = pd.read_csv(sequence_csv, keep_default_na=False)
    if "fruit_id" not in frame.columns or "crops_id" not in frame.columns:
        raise ValueError("sequence table must include fruit_id and crops_id")
    fruits = frame[_is_fruit(frame)].copy()
    fruits["predicted_fresh_weight_g"] = _predicted_weight(fruits)
    fruits["maturity_probability"] = (_numeric(fruits.get("stage_score", pd.Series([0] * len(fruits), index=fruits.index))) / 7.0).clip(0.0, 1.0)
    fruits["expected_harvest_weight_g"] = fruits["predicted_fresh_weight_g"] * fruits["maturity_probability"].clip(lower=0.25)

    group = (
        fruits.groupby("crops_id", dropna=False)
        .agg(
            fruit_count=("fruit_id", "nunique"),
            predicted_fresh_weight_g=("predicted_fresh_weight_g", "sum"),
            expected_harvest_weight_g=("expected_harvest_weight_g", "sum"),
            mean_maturity_probability=("maturity_probability", "mean"),
        )
        .reset_index()
    )
    fruit_output_csv.parent.mkdir(parents=True, exist_ok=True)
    group_output_csv.parent.mkdir(parents=True, exist_ok=True)
    fruits.to_csv(fruit_output_csv, index=False, encoding="utf-8")
    group.to_csv(group_output_csv, index=False, encoding="utf-8")
    return fruits, group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a reproducible baseline yield from sequence features.")
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--fruit-output", type=Path, default=Path("processed/yolo/fruit_yield_predictions.csv"))
    parser.add_argument("--group-output", type=Path, default=Path("processed/yolo/group_yield_predictions.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fruit_frame, group_frame = predict_yolo_kgcv_yield(args.sequence, args.fruit_output, args.group_output)
    print(f"wrote {len(fruit_frame)} fruit predictions and {len(group_frame)} group predictions")


if __name__ == "__main__":
    main()
