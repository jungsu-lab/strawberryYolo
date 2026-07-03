#!/usr/bin/env python3
"""Build a clean GitHub-ready package for the YOLO + KGCV pipeline."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PUBLIC_FILES = [
    "README.md",
    "requirements.txt",
    "LICENSE",
    "DATA_AND_MODEL_LICENSES.md",
    "GITHUB_RELEASE_CHECKLIST.md",
    "PUBLISH_TO_GITHUB.md",
    ".gitignore",
    "scripts/convert_aihub_strawberry_to_yolo.py",
    "scripts/build_yolo_label_sanity_report.py",
    "scripts/train_yolo26m_strawberry.py",
    "scripts/eval_yolo26m_strawberry.py",
    "scripts/yolo26m_infer_to_crops.py",
    "scripts/build_yolo_ablation_plan.py",
    "scripts/build_yolo_trait_table.py",
    "scripts/merge_trait_predictions.py",
    "scripts/build_gdd_features.py",
    "scripts/build_kgcv_fruit_sequences.py",
    "scripts/predict_yolo_kgcv_yield.py",
    "scripts/build_yolo_github_package.py",
    "tests/conftest.py",
    "tests/test_convert_aihub_strawberry_to_yolo.py",
    "tests/test_yolo_local_scripts.py",
    "tests/test_yolo_growth_pipeline_tables.py",
    "tests/test_build_yolo_github_package.py",
    "tests/test_build_yolo_ablation_plan.py",
    "reports/yolocsv/here.md",
    "reports/yolocsv/local_run_commands.md",
    ".github/workflows/yolo_pipeline_tests.yml",
]


def _copy_file(source_root: Path, output_root: Path, relative_path: str) -> bool:
    source = source_root / relative_path
    if not source.exists():
        return False
    target = output_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def _write_release_manifest(output_root: Path, copied: list[str], missing: list[str]) -> None:
    lines = [
        "# Release Manifest",
        "",
        "This package intentionally excludes raw data, processed data, model weights, runs, caches, and local virtual environments.",
        "",
        "## Copied Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in copied)
    if missing:
        lines.extend(["", "## Missing Optional Files", ""])
        lines.extend(f"- `{path}`" for path in missing)
    (output_root / "RELEASE_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_yolo_github_package(source_root: Path, output_root: Path, clean: bool = True) -> str:
    """Copy the public YOLO/KGCV pipeline files into a clean release directory."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []
    for relative_path in PUBLIC_FILES:
        if _copy_file(source_root, output_root, relative_path):
            copied.append(relative_path)
        else:
            missing.append(relative_path)

    _write_release_manifest(output_root, copied, missing)
    return "\n".join(copied)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean GitHub-ready YOLO/KGCV package.")
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("../strawberry_yolo_kgcv_github"))
    parser.add_argument("--no-clean", action="store_true", help="Do not delete output-root before copying.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copied = build_yolo_github_package(args.source_root, args.output_root, clean=not args.no_clean)
    copied_count = len([line for line in copied.splitlines() if line.strip()])
    print(f"copied {copied_count} files to {args.output_root}")


if __name__ == "__main__":
    main()
