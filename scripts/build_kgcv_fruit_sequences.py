#!/usr/bin/env python3
"""Build KGCV-style fruit sequence rows from YOLO trait and GDD tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STAGE_SCORE = {
    "잎": 1.0,
    "줄기": 1.0,
    "화방": 2.0,
    "꽃": 2.0,
    "과실": 5.0,
    "flower": 2.0,
    "small_g": 3.0,
    "green": 4.0,
    "white": 4.5,
    "turning_red": 5.0,
    "red": 6.0,
    "overripe": 7.0,
}


def _date_column(frame: pd.DataFrame) -> pd.Series:
    if "date_captured" in frame.columns:
        return pd.to_datetime(frame["date_captured"], errors="coerce").dt.date.astype(str)
    if "date" in frame.columns:
        return pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    raise ValueError("trait table must include date_captured or date")


def _ensure_fruit_id(frame: pd.DataFrame) -> pd.Series:
    if "fruit_id" in frame.columns:
        return frame["fruit_id"].astype(str)
    pieces = []
    for column in ["crops_id", "image_id", "annotation_id"]:
        if column in frame.columns:
            pieces.append(frame[column].astype(str))
    if not pieces:
        return pd.Series([f"row:{idx}" for idx in range(len(frame))], index=frame.index)
    fruit_id = pieces[0]
    for piece in pieces[1:]:
        fruit_id = fruit_id + ":" + piece
    return fruit_id


def build_kgcv_fruit_sequences(trait_csv: Path, gdd_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Merge trait rows with GDD and add sequence indices/deltas."""

    traits = pd.read_csv(trait_csv, keep_default_na=False)
    gdd = pd.read_csv(gdd_csv, keep_default_na=False)
    traits["fruit_id"] = _ensure_fruit_id(traits)
    traits["date"] = _date_column(traits)
    if "category_name" in traits.columns:
        traits["stage_score"] = traits["category_name"].map(STAGE_SCORE).fillna(0.0)
    elif "class_name" in traits.columns:
        traits["stage_score"] = traits["class_name"].map(STAGE_SCORE).fillna(0.0)
    else:
        traits["stage_score"] = 0.0

    merge_columns = ["crops_id", "date"]
    if "farm_id" in traits.columns and "farm_id" in gdd.columns:
        merge_columns = ["farm_id", "crops_id", "date"]
    merged = traits.merge(gdd, on=merge_columns, how="left", suffixes=("", "_gdd"))
    merged["gdd_cum"] = pd.to_numeric(merged.get("gdd_cum", 0.0), errors="coerce").fillna(0.0)
    merged = merged.sort_values(["fruit_id", "date", "image_id" if "image_id" in merged.columns else "fruit_id"]).reset_index(drop=True)
    merged["sequence_index"] = merged.groupby("fruit_id", dropna=False).cumcount()
    merged["delta_gdd"] = merged.groupby("fruit_id", dropna=False)["gdd_cum"].diff().fillna(0.0)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False, encoding="utf-8")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fruit sequence rows for KGCV-style growth modeling.")
    parser.add_argument("--trait-table", type=Path, required=True)
    parser.add_argument("--gdd", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("processed/yolo/fruit_sequences.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_kgcv_fruit_sequences(args.trait_table, args.gdd, args.output)
    print(f"wrote {len(frame)} sequence rows to {args.output}")


if __name__ == "__main__":
    main()
