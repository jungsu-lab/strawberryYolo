#!/usr/bin/env python3
"""Merge KGCV or external trait model predictions into the object trait table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_TRAIT_COLUMNS = ["diameter_mm", "length_mm", "decimal_stage", "trait_model"]


def merge_trait_predictions(
    trait_table_csv: Path,
    trait_predictions_csv: Path,
    output_csv: Path,
    key: str = "fruit_id",
) -> pd.DataFrame:
    """Merge predicted diameter/length/stage columns into a trait table."""

    trait_table = pd.read_csv(trait_table_csv, keep_default_na=False)
    predictions = pd.read_csv(trait_predictions_csv, keep_default_na=False)
    if key not in trait_table.columns:
        raise ValueError(f"trait table missing merge key: {key}")
    if key not in predictions.columns:
        raise ValueError(f"trait predictions missing merge key: {key}")

    prediction_columns = [key] + [column for column in DEFAULT_TRAIT_COLUMNS if column in predictions.columns]
    if len(prediction_columns) == 1:
        raise ValueError(f"trait predictions must include at least one of {DEFAULT_TRAIT_COLUMNS}")
    merged = trait_table.merge(predictions[prediction_columns], on=key, how="left")
    for column in ["diameter_mm", "length_mm", "decimal_stage"]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False, encoding="utf-8")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge KGCV/external trait predictions into trait_table.csv.")
    parser.add_argument("--trait-table", type=Path, required=True)
    parser.add_argument("--trait-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("processed/yolo/trait_table_with_predictions.csv"))
    parser.add_argument("--key", default="fruit_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = merge_trait_predictions(args.trait_table, args.trait_predictions, args.output, args.key)
    print(f"wrote {len(frame)} merged trait rows to {args.output}")


if __name__ == "__main__":
    main()
