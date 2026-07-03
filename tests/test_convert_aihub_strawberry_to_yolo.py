import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.convert_aihub_strawberry_to_yolo import convert_aihub_strawberry_to_yolo


def _write_sample_label(path: Path, fname: str = "sample.jpg") -> None:
    payload = {
        "images": {
            "image_id": 32435,
            "farm_id": "AI22_002",
            "crops_id": "BERRY_13_001",
            "crops": "딸기",
            "kind_type": "설향",
            "file_path": "/원천데이터/01.딸기/001.설향/01.정상/",
            "fname": fname,
            "fext": "jpg",
            "width": "1071",
            "height": "1064",
            "create_date": "2022-09-26",
            "date_captured": "2022-09-09 09:17:38",
            "growth_stage": "정식기",
            "leaf": "1",
            "disease_class": "정상",
            "disease_cause": "정상",
            "disease_cause_method": "정상",
            "plant_body": "0",
        },
        "environments": [
            {
                "farm_id": "AI22_002",
                "receive_date": "2022-09-01 17:26:05",
                "id": 0,
                "ti_value": "20.0",
                "hi_value": "80.0",
                "ci_value": None,
                "tl_value": "22.1",
            },
            {
                "farm_id": "AI22_002",
                "receive_date": "2022-09-01 17:27:05",
                "id": 1,
                "ti_value": "22.0",
                "hi_value": "82.0",
                "ci_value": None,
                "tl_value": "22.3",
            },
        ],
        "growth_index": {
            "crops_id": "BERRY_13_001",
            "measured_date": "2022-09-13",
            "stem_length": "26",
            "leaf_cnt": "6",
            "leaf_width": "7",
            "leaf_length": "8.8",
            "stem_thick": "13.8",
            "fr1_cnt": None,
            "fr_weight": None,
        },
        "annotations": [
            {
                "bbox": [209.4, 495.72, 201.25, 114.2],
                "id": 0,
                "image_id": 32435,
                "category_id": 0,
                "disease_status": "N",
            },
            {
                "bbox": [480.57, 673.39, 172.11, 94.92],
                "id": 2,
                "image_id": 32435,
                "category_id": 2,
                "disease_status": "N",
            },
        ],
        "categories": [
            {"id": 0, "name": "잎", "supercategory": "none"},
            {"id": 2, "name": "줄기", "supercategory": "none"},
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_convert_aihub_json_to_yolo_dataset_and_metadata(tmp_path):
    label_path = tmp_path / "labels" / "sample.json"
    label_path.parent.mkdir()
    _write_sample_label(label_path)
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "sample.jpg").write_bytes(b"fake-image")
    output_dir = tmp_path / "yolo"

    result = convert_aihub_strawberry_to_yolo(
        label_inputs=[label_path],
        image_root=image_root,
        output_dir=output_dir,
        class_names=["화방", "과실", "줄기", "잎"],
        split_ratios=(1.0, 0.0, 0.0),
        image_policy="symlink",
    )

    assert result.image_count == 1
    assert result.annotation_count == 2
    label_lines = (output_dir / "labels" / "train" / "sample.txt").read_text(encoding="utf-8").splitlines()
    assert len(label_lines) == 2
    assert label_lines[0].startswith("3 ")
    assert label_lines[1].startswith("2 ")

    first = [float(value) for value in label_lines[0].split()[1:]]
    assert first[0] == pytest_approx((209.4 + 201.25 / 2) / 1071)
    assert first[1] == pytest_approx((495.72 + 114.2 / 2) / 1064)
    assert first[2] == pytest_approx(201.25 / 1071)
    assert first[3] == pytest_approx(114.2 / 1064)

    data_yaml = yaml.safe_load((output_dir / "aihub_strawberry_yolo.yaml").read_text(encoding="utf-8"))
    assert data_yaml["names"] == {0: "화방", 1: "과실", 2: "줄기", 3: "잎"}
    assert data_yaml["train"] == "images/train"

    manifest = pd.read_csv(output_dir / "metadata" / "annotation_manifest.csv", keep_default_na=False)
    assert manifest["category_name"].tolist() == ["잎", "줄기"]
    assert manifest["growth_stage"].tolist() == ["정식기", "정식기"]
    assert manifest["image_width"].tolist() == [1071.0, 1071.0]
    assert manifest["image_height"].tolist() == [1064.0, 1064.0]

    features = pd.read_csv(output_dir / "metadata" / "image_growth_features.csv", keep_default_na=False)
    row = features.iloc[0]
    assert row["image_id"] == 32435
    assert row["crops_id"] == "BERRY_13_001"
    assert float(row["growth_leaf_cnt"]) == 6
    assert float(row["env_ti_value_mean"]) == 21.0
    assert float(row["env_tl_value_latest"]) == 22.3


def test_converter_clips_boxes_and_records_invalid_annotations(tmp_path):
    label_path = tmp_path / "label.json"
    _write_sample_label(label_path, fname="edge.jpg")
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    payload["annotations"] = [
        {"bbox": [-5, 10, 20, 20], "id": 1, "image_id": 32435, "category_id": 0},
        {"bbox": [10, 10, 0, 20], "id": 2, "image_id": 32435, "category_id": 999},
    ]
    label_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "edge.jpg").write_bytes(b"fake-image")

    result = convert_aihub_strawberry_to_yolo(
        label_inputs=[label_path],
        image_root=image_root,
        output_dir=tmp_path / "out",
        class_names=["잎"],
        split_ratios=(1.0, 0.0, 0.0),
        image_policy="copy",
    )

    assert result.annotation_count == 1
    label_line = (tmp_path / "out" / "labels" / "train" / "edge.txt").read_text(encoding="utf-8").strip()
    assert label_line.startswith("0 ")
    errors = pd.read_csv(tmp_path / "out" / "metadata" / "conversion_errors.csv", keep_default_na=False)
    assert errors["reason"].tolist() == ["invalid_bbox"]


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-6)
