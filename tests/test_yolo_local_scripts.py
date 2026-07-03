from pathlib import Path
import builtins

import pandas as pd
import pytest
import yaml
from PIL import Image

from scripts.build_yolo_label_sanity_report import build_yolo_label_sanity_report
from scripts.eval_yolo26m_strawberry import build_val_config, eval_yolo26m_strawberry
from scripts.train_yolo26m_strawberry import build_train_config, train_yolo26m_strawberry
from scripts.yolo26m_infer_to_crops import crop_with_margin, run_yolo_inference_to_crops


def test_build_yolo_label_sanity_report_draws_sample_image(tmp_path):
    dataset_dir = tmp_path / "dataset"
    image_dir = dataset_dir / "images" / "train"
    label_dir = dataset_dir / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (100, 80), "white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("0 0.50000000 0.50000000 0.40000000 0.50000000\n", encoding="utf-8")
    (dataset_dir / "aihub_strawberry_yolo.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "과실"},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "sanity"

    summary = build_yolo_label_sanity_report(
        dataset_yaml=dataset_dir / "aihub_strawberry_yolo.yaml",
        output_dir=output_dir,
        splits=["train"],
        sample_count=1,
    )

    assert summary.image_count == 1
    assert summary.box_count == 1
    assert (output_dir / "train" / "sample.jpg").exists()
    summary_csv = pd.read_csv(output_dir / "label_sanity_summary.csv", keep_default_na=False)
    assert summary_csv.iloc[0]["split"] == "train"
    assert summary_csv.iloc[0]["box_count"] == 1


def test_crop_with_margin_returns_fixed_square_crop():
    image = Image.new("RGB", (100, 80), "white")

    crop = crop_with_margin(image, [40, 30, 60, 50], margin=0.5, output_size=256)

    assert crop.size == (256, 256)


def test_build_train_config_is_dry_run_safe(tmp_path):
    config = build_train_config(
        data_yaml=tmp_path / "dataset.yaml",
        model="yolo26m.pt",
        imgsz=1280,
        epochs=120,
        batch=-1,
        device="0",
        project=tmp_path / "runs",
        name="strawberry_stage_det",
        patience=30,
    )

    assert config["data"] == str(tmp_path / "dataset.yaml")
    assert config["model"] == "yolo26m.pt"
    assert config["imgsz"] == 1280
    assert config["project"] == str(tmp_path / "runs")


def test_build_val_config_is_dry_run_safe(tmp_path):
    config = build_val_config(
        data_yaml=tmp_path / "dataset.yaml",
        model_path=tmp_path / "best.pt",
        imgsz=1280,
        batch=8,
        device="0",
        project=tmp_path / "runs",
        name="val",
    )

    assert config["data"] == str(tmp_path / "dataset.yaml")
    assert config["model"] == str(tmp_path / "best.pt")
    assert config["imgsz"] == 1280
    assert config["batch"] == 8


def _block_ultralytics_import(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ModuleNotFoundError("No module named 'ultralytics'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_train_reports_clear_message_when_ultralytics_is_missing(tmp_path, monkeypatch):
    _block_ultralytics_import(monkeypatch)
    config = build_train_config(data_yaml=tmp_path / "dataset.yaml")

    with pytest.raises(SystemExit, match="pip install -r requirements.txt"):
        train_yolo26m_strawberry(config, dry_run=False)


def test_eval_reports_clear_message_when_ultralytics_is_missing(tmp_path, monkeypatch):
    _block_ultralytics_import(monkeypatch)
    config = build_val_config(data_yaml=tmp_path / "dataset.yaml", model_path=tmp_path / "best.pt")

    with pytest.raises(SystemExit, match="pip install -r requirements.txt"):
        eval_yolo26m_strawberry(config, dry_run=False)


def test_infer_reports_clear_message_when_ultralytics_is_missing(tmp_path, monkeypatch):
    _block_ultralytics_import(monkeypatch)

    with pytest.raises(SystemExit, match="pip install -r requirements.txt"):
        run_yolo_inference_to_crops(
            model_path=tmp_path / "best.pt",
            source=tmp_path / "images",
            output_dir=tmp_path / "crops",
        )
