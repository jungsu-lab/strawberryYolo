import pandas as pd

from scripts.build_yolo_ablation_plan import build_yolo_ablation_plan


def test_build_yolo_ablation_plan_writes_train_and_eval_commands(tmp_path):
    output_csv = tmp_path / "ablation_plan.csv"

    frame = build_yolo_ablation_plan(
        data_yaml=tmp_path / "dataset.yaml",
        output_csv=output_csv,
        models=["yolo26n.pt", "yolo26m.pt"],
        image_sizes=[640, 1280],
        conf_thresholds=[0.1, 0.2],
        epochs=50,
        batch=-1,
        device="0",
        project="runs/yolo_ablation",
    )

    assert len(frame) == 8
    assert output_csv.exists()
    saved = pd.read_csv(output_csv, keep_default_na=False)
    first = saved.iloc[0]
    assert first["experiment_id"] == "yolo26n_img640_conf010"
    assert "--imgsz 640" in first["train_command"]
    assert "--epochs 50" in first["train_command"]
    assert "--dry-run" not in first["train_command"]
    assert "--conf 0.1" in first["crop_command"]
    assert "weights/best.pt" in first["eval_command"]
