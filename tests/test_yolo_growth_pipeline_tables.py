from pathlib import Path

import pandas as pd

from scripts.build_gdd_features import build_gdd_features
from scripts.build_kgcv_fruit_sequences import build_kgcv_fruit_sequences
from scripts.build_yolo_trait_table import build_yolo_trait_table
from scripts.merge_trait_predictions import merge_trait_predictions
from scripts.predict_yolo_kgcv_yield import predict_yolo_kgcv_yield


def test_build_yolo_trait_table_merges_annotations_and_growth_features(tmp_path):
    annotation_manifest = tmp_path / "annotation_manifest.csv"
    image_features = tmp_path / "image_growth_features.csv"
    annotation_manifest.write_text(
        "image_id,fname,split,farm_id,crops_id,growth_stage,date_captured,annotation_id,category_name,class_id,bbox_x,bbox_y,bbox_w,bbox_h,image_width,image_height\n"
        "1,a.jpg,train,F1,C1,수확기,2022-12-05 13:00:00,0,과실,1,10,20,30,40,100,200\n"
        "1,a.jpg,train,F1,C1,수확기,2022-12-05 13:00:00,1,줄기,2,20,30,10,50,100,200\n",
        encoding="utf-8",
    )
    image_features.write_text(
        "image_id,farm_id,crops_id,date_captured,growth_leaf_cnt,growth_stem_length,env_ti_value_mean\n"
        "1,F1,C1,2022-12-05 13:00:00,8,25.5,21.2\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "traits.csv"

    frame = build_yolo_trait_table(annotation_manifest, image_features, output_csv)

    assert output_csv.exists()
    assert len(frame) == 2
    fruit = frame[frame["category_name"] == "과실"].iloc[0]
    assert fruit["fruit_id"] == "C1:1:0"
    assert fruit["bbox_area_px"] == 1200.0
    assert fruit["bbox_area_ratio"] == 0.06
    assert fruit["bbox_aspect_ratio"] == 0.75
    assert fruit["growth_leaf_cnt"] == 8
    assert fruit["env_ti_value_mean"] == 21.2


def test_build_gdd_features_uses_temperature_and_accumulates_by_crop(tmp_path):
    image_features = tmp_path / "image_growth_features.csv"
    image_features.write_text(
        "image_id,farm_id,crops_id,date_captured,env_ti_value_mean\n"
        "1,F1,C1,2022-12-05 13:00:00,21\n"
        "2,F1,C1,2022-12-06 13:00:00,23\n"
        "3,F1,C2,2022-12-05 13:00:00,19\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "gdd.csv"

    frame = build_gdd_features(image_features, output_csv, base_temperature=10.0)

    c1 = frame[frame["crops_id"] == "C1"].sort_values("date")
    assert c1["gdd_day"].tolist() == [11.0, 13.0]
    assert c1["gdd_cum"].tolist() == [11.0, 24.0]
    assert output_csv.exists()


def test_merge_trait_predictions_adds_kgcv_outputs_by_fruit_id(tmp_path):
    trait_table = tmp_path / "trait_table.csv"
    predictions = tmp_path / "trait_predictions.csv"
    output_csv = tmp_path / "trait_table_with_predictions.csv"
    trait_table.write_text(
        "fruit_id,crops_id,category_name,bbox_w,bbox_h\n"
        "F1,C1,과실,30,40\n"
        "F2,C1,줄기,10,50\n",
        encoding="utf-8",
    )
    predictions.write_text(
        "fruit_id,diameter_mm,length_mm,decimal_stage,trait_model\n"
        "F1,18.5,22.0,5.2,kgcv_cnn\n",
        encoding="utf-8",
    )

    frame = merge_trait_predictions(trait_table, predictions, output_csv)

    assert frame.loc[frame["fruit_id"] == "F1", "diameter_mm"].iloc[0] == 18.5
    assert frame.loc[frame["fruit_id"] == "F1", "trait_model"].iloc[0] == "kgcv_cnn"
    assert pd.isna(frame.loc[frame["fruit_id"] == "F2", "diameter_mm"]).iloc[0]
    assert output_csv.exists()


def test_build_kgcv_fruit_sequences_merges_gdd_and_computes_delta(tmp_path):
    trait_csv = tmp_path / "traits.csv"
    gdd_csv = tmp_path / "gdd.csv"
    trait_csv.write_text(
        "fruit_id,crops_id,date_captured,category_name,bbox_w,bbox_h,bbox_area_ratio,growth_leaf_cnt\n"
        "C1:1:0,C1,2022-12-05 13:00:00,과실,30,40,0.06,8\n"
        "C1:1:0,C1,2022-12-06 13:00:00,과실,32,42,0.07,8\n",
        encoding="utf-8",
    )
    gdd_csv.write_text(
        "farm_id,crops_id,date,temp_mean,gdd_day,gdd_cum\n"
        "F1,C1,2022-12-05,21,11,11\n"
        "F1,C1,2022-12-06,23,13,24\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "sequence.csv"

    frame = build_kgcv_fruit_sequences(trait_csv, gdd_csv, output_csv)

    assert frame["sequence_index"].tolist() == [0, 1]
    assert frame["delta_gdd"].tolist() == [0.0, 13.0]
    assert frame["stage_score"].tolist() == [5.0, 5.0]
    assert output_csv.exists()


def test_predict_yolo_kgcv_yield_outputs_fruit_and_group_predictions(tmp_path):
    sequence_csv = tmp_path / "sequence.csv"
    fruit_output = tmp_path / "fruit_yield.csv"
    group_output = tmp_path / "group_yield.csv"
    sequence_csv.write_text(
        "fruit_id,crops_id,date,category_name,bbox_w,bbox_h,bbox_area_ratio,stage_score,growth_fr_weight\n"
        "F1,C1,2022-12-05,과실,30,40,0.06,5,\n"
        "F2,C1,2022-12-05,과실,20,30,0.03,5,12.5\n"
        "S1,C1,2022-12-05,줄기,10,50,0.02,1,\n",
        encoding="utf-8",
    )

    fruit_frame, group_frame = predict_yolo_kgcv_yield(sequence_csv, fruit_output, group_output)

    assert fruit_frame["fruit_id"].tolist() == ["F1", "F2"]
    assert fruit_frame.loc[fruit_frame["fruit_id"] == "F2", "predicted_fresh_weight_g"].iloc[0] == 12.5
    assert group_frame.iloc[0]["crops_id"] == "C1"
    assert group_frame.iloc[0]["fruit_count"] == 2
    assert fruit_output.exists()
    assert group_output.exists()
