from pathlib import Path

from scripts.build_yolo_github_package import build_yolo_github_package


def test_build_yolo_github_package_copies_only_public_pipeline_files(tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "release"
    (source_root / "scripts").mkdir(parents=True)
    (source_root / "tests").mkdir()
    (source_root / "reports" / "yolocsv").mkdir(parents=True)
    (source_root / ".github" / "workflows").mkdir(parents=True)
    (source_root / "raw").mkdir()
    (source_root / "processed").mkdir()

    for path in [
        "README.md",
        "requirements.txt",
        "LICENSE",
        "DATA_AND_MODEL_LICENSES.md",
        "GITHUB_RELEASE_CHECKLIST.md",
        "PUBLISH_TO_GITHUB.md",
        ".gitignore",
        "AGENTS.md",
        "scripts/convert_aihub_strawberry_to_yolo.py",
        "scripts/build_yolo_trait_table.py",
        "tests/test_convert_aihub_strawberry_to_yolo.py",
        "reports/yolocsv/local_run_commands.md",
        ".github/workflows/yolo_pipeline_tests.yml",
    ]:
        file_path = source_root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(path, encoding="utf-8")

    (source_root / "raw" / "secret.zip").write_text("do not copy", encoding="utf-8")
    (source_root / "processed" / "big.csv").write_text("do not copy", encoding="utf-8")

    manifest = build_yolo_github_package(source_root, output_root)

    assert (output_root / "README.md").exists()
    assert (output_root / "LICENSE").exists()
    assert (output_root / "DATA_AND_MODEL_LICENSES.md").exists()
    assert (output_root / "PUBLISH_TO_GITHUB.md").exists()
    assert (output_root / "scripts" / "convert_aihub_strawberry_to_yolo.py").exists()
    assert (output_root / "tests" / "test_convert_aihub_strawberry_to_yolo.py").exists()
    assert (output_root / ".github" / "workflows" / "yolo_pipeline_tests.yml").exists()
    assert not (output_root / "AGENTS.md").exists()
    assert not (output_root / "raw").exists()
    assert not (output_root / "processed").exists()
    assert (output_root / "RELEASE_MANIFEST.md").exists()
    assert "scripts/convert_aihub_strawberry_to_yolo.py" in manifest
