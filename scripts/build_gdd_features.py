#!/usr/bin/env python3
"""Build simple GDD features from converted image-level environment metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_TEMPERATURE_COLUMNS = [
    "env_ti_value_mean",
    "env_tl_value_mean",
    "env_cl_value_mean",
]


def _choose_temperature_column(frame: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"temperature column not found: {requested}")
        return requested
    for column in DEFAULT_TEMPERATURE_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(f"no supported temperature column found; tried {DEFAULT_TEMPERATURE_COLUMNS}")


def build_gdd_features(
    image_growth_features_csv: Path,
    output_csv: Path,
    base_temperature: float = 10.0,
    temperature_column: str | None = None,
) -> pd.DataFrame:
    """Aggregate image-level temperature by date and compute cumulative GDD."""

    frame = pd.read_csv(image_growth_features_csv, keep_default_na=False)
    if "date_captured" not in frame.columns or "crops_id" not in frame.columns:
        raise ValueError("image growth features must include date_captured and crops_id")
    temp_column = _choose_temperature_column(frame, temperature_column)
    frame["date"] = pd.to_datetime(frame["date_captured"], errors="coerce").dt.date.astype(str)
    frame["temp_mean"] = pd.to_numeric(frame[temp_column], errors="coerce")
    frame = frame.dropna(subset=["date", "temp_mean"])
    group_columns = [column for column in ["farm_id", "crops_id", "date"] if column in frame.columns]
    daily = frame.groupby(group_columns, dropna=False, as_index=False)["temp_mean"].mean()
    daily["gdd_day"] = (daily["temp_mean"] - base_temperature).clip(lower=0.0)
    sort_columns = [column for column in ["farm_id", "crops_id", "date"] if column in daily.columns]
    daily = daily.sort_values(sort_columns).reset_index(drop=True)
    cumulative_groups = [column for column in ["farm_id", "crops_id"] if column in daily.columns]
    daily["gdd_cum"] = daily.groupby(cumulative_groups, dropna=False)["gdd_day"].cumsum()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_csv, index=False, encoding="utf-8")
    return daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily GDD features from image-level environment metadata.")
    parser.add_argument("--image-growth-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("processed/yolo/gdd_daily.csv"))
    parser.add_argument("--base-temperature", type=float, default=10.0)
    parser.add_argument("--temperature-column", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_gdd_features(args.image_growth_features, args.output, args.base_temperature, args.temperature_column)
    print(f"wrote {len(frame)} GDD rows to {args.output}")


if __name__ == "__main__":
    main()
