#!/usr/bin/env python3
"""Convert AIHub-style strawberry annotation JSON files into a YOLO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


DEFAULT_CLASS_NAMES = ["화방", "과실", "줄기", "잎"]
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ConversionResult:
    image_count: int
    annotation_count: int
    skipped_annotation_count: int
    skipped_image_count: int
    class_names: list[str]
    output_dir: Path


def _as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _label_files(label_inputs: Iterable[Path], max_files: int | None = None) -> list[Path]:
    paths: list[Path] = []
    for input_path in label_inputs:
        if input_path.is_dir():
            paths.extend(sorted(path for path in input_path.rglob("*.json") if path.is_file()))
        elif input_path.is_file():
            paths.append(input_path)
        else:
            raise FileNotFoundError(f"label input does not exist: {input_path}")
    paths = sorted(dict.fromkeys(paths))
    if max_files is not None and max_files >= 0:
        paths = paths[:max_files]
    return paths


def _category_lookup(categories: list[dict[str, Any]]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for category in categories:
        category_id = _as_int(category.get("id"))
        name = str(category.get("name", "")).strip()
        if category_id is not None and name:
            lookup[category_id] = name
    return lookup


def _image_stem(fname: str) -> str:
    return Path(fname).stem


def _split_for_key(key: str, split_ratios: tuple[float, float, float], seed: int) -> str:
    train_ratio, val_ratio, _test_ratio = split_ratios
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def _validate_split_ratios(split_ratios: tuple[float, float, float]) -> None:
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must contain train, val, test values")
    if any(value < 0 for value in split_ratios):
        raise ValueError("split ratios must be non-negative")
    total = sum(split_ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")


def _resolve_image_path(image_root: Path, image_info: dict[str, Any]) -> Path | None:
    fname = str(image_info.get("fname", "")).strip()
    if not fname:
        return None
    relative_dir = str(image_info.get("file_path", "")).strip().lstrip("/")
    candidates = []
    if relative_dir:
        candidates.append(image_root / relative_dir / fname)
    candidates.append(image_root / fname)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _materialize_image(source_path: Path, target_path: Path, image_policy: str) -> None:
    if image_policy == "none":
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    if image_policy == "copy":
        shutil.copy2(source_path, target_path)
        return
    if image_policy == "symlink":
        try:
            target_path.symlink_to(source_path.resolve())
        except OSError:
            shutil.copy2(source_path, target_path)
        return
    raise ValueError(f"unsupported image_policy: {image_policy}")


def _clip_bbox_xywh(bbox: list[Any], width: float, height: float) -> tuple[float, float, float, float] | None:
    if len(bbox) != 4:
        return None
    x, y, w, h = (_as_float(value) for value in bbox)
    if x is None or y is None or w is None or h is None:
        return None
    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return None
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    return x_center, y_center, clipped_w / width, clipped_h / height


def _format_yolo_line(class_id: int, normalized_bbox: tuple[float, float, float, float]) -> str:
    values = " ".join(f"{value:.8f}" for value in normalized_bbox)
    return f"{class_id} {values}"


def _numeric_env_summary(environments: list[dict[str, Any]]) -> dict[str, float | str]:
    summary: dict[str, float | str] = {}
    if not environments:
        return summary
    sorted_env = sorted(environments, key=lambda row: str(row.get("receive_date", "")))
    numeric_keys = sorted(
        {
            key
            for row in sorted_env
            for key, value in row.items()
            if key.endswith("_value") and _as_float(value) is not None
        }
    )
    for key in numeric_keys:
        values = [_as_float(row.get(key)) for row in sorted_env]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            continue
        latest = next((_as_float(row.get(key)) for row in reversed(sorted_env) if _as_float(row.get(key)) is not None), None)
        summary[f"env_{key}_mean"] = sum(numeric_values) / len(numeric_values)
        summary[f"env_{key}_min"] = min(numeric_values)
        summary[f"env_{key}_max"] = max(numeric_values)
        if latest is not None:
            summary[f"env_{key}_latest"] = latest
    summary["env_record_count"] = len(environments)
    return summary


def _growth_features(growth_index: dict[str, Any]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for key, value in growth_index.items():
        output_key = f"growth_{key}"
        numeric = _as_float(value)
        features[output_key] = numeric if numeric is not None else ("" if value is None else value)
    return features


def _write_yaml(output_dir: Path, class_names: list[str]) -> None:
    payload = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    (output_dir / "aihub_strawberry_yolo.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_summary(output_dir: Path, result: ConversionResult, split_counts: dict[str, int]) -> None:
    rows = [
        "# AIHub Strawberry YOLO Conversion Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| image_count | {result.image_count} |",
        f"| annotation_count | {result.annotation_count} |",
        f"| skipped_annotation_count | {result.skipped_annotation_count} |",
        f"| skipped_image_count | {result.skipped_image_count} |",
        "",
        "## Split Counts",
        "",
        "| split | image_count |",
        "|---|---:|",
    ]
    rows.extend(f"| {split} | {split_counts.get(split, 0)} |" for split in SPLITS)
    rows.extend(
        [
            "",
            "## Class Names",
            "",
        ]
    )
    rows.extend(f"- {idx}: {name}" for idx, name in enumerate(result.class_names))
    (output_dir / "metadata" / "conversion_summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def convert_aihub_strawberry_to_yolo(
    label_inputs: list[Path],
    image_root: Path,
    output_dir: Path,
    class_names: list[str] | None = None,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    image_policy: str = "symlink",
    seed: int = 42,
    max_files: int | None = None,
    dry_run: bool = False,
) -> ConversionResult:
    """Convert AIHub-style JSON labels into a YOLO train/val/test folder."""

    _validate_split_ratios(split_ratios)
    class_names = class_names or DEFAULT_CLASS_NAMES
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    label_files = _label_files(label_inputs, max_files=max_files)

    manifest_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    split_counts = {split: 0 for split in SPLITS}
    valid_annotation_count = 0
    skipped_image_count = 0

    if not dry_run:
        for split in SPLITS:
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata").mkdir(parents=True, exist_ok=True)

    for label_path in label_files:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        image_info = payload.get("images") or {}
        image_id = str(image_info.get("image_id", label_path.stem))
        fname = str(image_info.get("fname", f"{label_path.stem}.jpg")).strip()
        width = _as_float(image_info.get("width"))
        height = _as_float(image_info.get("height"))
        if width is None or height is None or width <= 0 or height <= 0:
            error_rows.append({"label_path": str(label_path), "image_id": image_id, "annotation_id": "", "reason": "invalid_image_size"})
            skipped_image_count += 1
            continue

        source_image = _resolve_image_path(image_root, image_info)
        if source_image is None and image_policy != "none":
            error_rows.append({"label_path": str(label_path), "image_id": image_id, "annotation_id": "", "reason": "missing_image"})
            skipped_image_count += 1
            continue

        split = _split_for_key(image_id or fname, split_ratios, seed)
        stem = _image_stem(fname)
        yolo_lines: list[str] = []
        categories = _category_lookup(payload.get("categories") or [])

        for annotation in payload.get("annotations") or []:
            annotation_id = annotation.get("id", "")
            normalized_bbox = _clip_bbox_xywh(annotation.get("bbox") or [], width, height)
            if normalized_bbox is None:
                error_rows.append(
                    {"label_path": str(label_path), "image_id": image_id, "annotation_id": annotation_id, "reason": "invalid_bbox"}
                )
                continue
            category_id = _as_int(annotation.get("category_id"))
            category_name = categories.get(category_id if category_id is not None else -1, "")
            if category_name not in class_to_id:
                error_rows.append(
                    {
                        "label_path": str(label_path),
                        "image_id": image_id,
                        "annotation_id": annotation_id,
                        "reason": "unknown_category",
                    }
                )
                continue
            class_id = class_to_id[category_name]
            yolo_lines.append(_format_yolo_line(class_id, normalized_bbox))
            valid_annotation_count += 1
            manifest_rows.append(
                {
                    "label_path": str(label_path),
                    "image_id": image_id,
                    "fname": fname,
                    "split": split,
                    "farm_id": image_info.get("farm_id", ""),
                    "crops_id": image_info.get("crops_id", ""),
                    "growth_stage": image_info.get("growth_stage", ""),
                    "date_captured": image_info.get("date_captured", ""),
                    "image_width": width,
                    "image_height": height,
                    "annotation_id": annotation_id,
                    "category_id": category_id if category_id is not None else "",
                    "category_name": category_name,
                    "class_id": class_id,
                    "disease_status": annotation.get("disease_status", ""),
                    "bbox_x": annotation.get("bbox", ["", "", "", ""])[0],
                    "bbox_y": annotation.get("bbox", ["", "", "", ""])[1],
                    "bbox_w": annotation.get("bbox", ["", "", "", ""])[2],
                    "bbox_h": annotation.get("bbox", ["", "", "", ""])[3],
                }
            )

        if not dry_run:
            label_output = output_dir / "labels" / split / f"{stem}.txt"
            label_output.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            if source_image is not None:
                _materialize_image(source_image, output_dir / "images" / split / fname, image_policy)
        split_counts[split] += 1

        feature_row = {
            "label_path": str(label_path),
            "image_id": _as_int(image_info.get("image_id")) or image_info.get("image_id", ""),
            "fname": fname,
            "split": split,
            "farm_id": image_info.get("farm_id", ""),
            "crops_id": image_info.get("crops_id", ""),
            "crops": image_info.get("crops", ""),
            "kind_type": image_info.get("kind_type", ""),
            "growth_stage": image_info.get("growth_stage", ""),
            "date_captured": image_info.get("date_captured", ""),
            "create_date": image_info.get("create_date", ""),
            "disease_class": image_info.get("disease_class", ""),
            "leaf": _as_float(image_info.get("leaf")) if _as_float(image_info.get("leaf")) is not None else image_info.get("leaf", ""),
            "plant_body": _as_float(image_info.get("plant_body")) if _as_float(image_info.get("plant_body")) is not None else image_info.get("plant_body", ""),
        }
        feature_row.update(_growth_features(payload.get("growth_index") or {}))
        feature_row.update(_numeric_env_summary(payload.get("environments") or []))
        feature_rows.append(feature_row)

    result = ConversionResult(
        image_count=sum(split_counts.values()),
        annotation_count=valid_annotation_count,
        skipped_annotation_count=len([row for row in error_rows if row.get("annotation_id") != ""]),
        skipped_image_count=skipped_image_count,
        class_names=class_names,
        output_dir=output_dir,
    )

    if not dry_run:
        metadata_dir = output_dir / "metadata"
        pd.DataFrame(manifest_rows).to_csv(metadata_dir / "annotation_manifest.csv", index=False, encoding="utf-8")
        pd.DataFrame(feature_rows).to_csv(metadata_dir / "image_growth_features.csv", index=False, encoding="utf-8")
        pd.DataFrame(error_rows, columns=["label_path", "image_id", "annotation_id", "reason"]).to_csv(
            metadata_dir / "conversion_errors.csv",
            index=False,
            encoding="utf-8",
        )
        if manifest_rows:
            class_distribution = (
                pd.DataFrame(manifest_rows)
                .groupby(["class_id", "category_name"], dropna=False)
                .size()
                .reset_index(name="annotation_count")
                .sort_values(["class_id", "category_name"])
            )
        else:
            class_distribution = pd.DataFrame(columns=["class_id", "category_name", "annotation_count"])
        class_distribution.to_csv(metadata_dir / "class_distribution.csv", index=False, encoding="utf-8")
        _write_yaml(output_dir, class_names)
        _write_summary(output_dir, result, split_counts)
    return result


def _parse_split(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--split must be formatted like 0.8,0.1,0.1")
    return parts[0], parts[1], parts[2]


def _parse_class_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("--class-names must include at least one class")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AIHub strawberry JSON labels to YOLO format.")
    parser.add_argument("label_inputs", nargs="+", type=Path, help="JSON file(s) or directory/directories containing JSON labels.")
    parser.add_argument("--image-root", type=Path, required=True, help="Root folder containing image files.")
    parser.add_argument("--output-dir", type=Path, default=Path("processed/yolo/aihub_strawberry"))
    parser.add_argument("--class-names", type=_parse_class_names, default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--split", type=_parse_split, default=(0.8, 0.1, 0.1), help="Train,val,test ratios, e.g. 0.8,0.1,0.1.")
    parser.add_argument("--image-policy", choices=["symlink", "copy", "none"], default="symlink")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None, help="Safety limit for smoke tests. Omit for all explicitly provided labels.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and validate without writing output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = convert_aihub_strawberry_to_yolo(
        label_inputs=args.label_inputs,
        image_root=args.image_root,
        output_dir=args.output_dir,
        class_names=args.class_names,
        split_ratios=args.split,
        image_policy=args.image_policy,
        seed=args.seed,
        max_files=args.max_files,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "image_count": result.image_count,
                "annotation_count": result.annotation_count,
                "skipped_annotation_count": result.skipped_annotation_count,
                "skipped_image_count": result.skipped_image_count,
                "output_dir": str(result.output_dir),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
