# Strawberry YOLO26m + KGCV Growth Prediction Pipeline

AIHub형 딸기 생육 JSON 라벨을 YOLO 학습 데이터로 변환하고, YOLO26m detector 결과를 KGCV-style 생장/수량 예측 파이프라인으로 연결하기 위한 로컬 실행 프로젝트입니다.

현재 구현은 GitHub에 올려 재현 가능한 MVP를 목표로 합니다. 실제 KGCV pretrained CNN/growthNet 연결 전에도 다음 단계가 동작합니다.

```text
AIHub JSON labels
-> YOLO train/val/test dataset
-> label sanity images
-> YOLO26m train/eval wrapper
-> YOLO inference crops
-> object trait table
-> GDD table
-> fruit sequence table
-> yield baseline table
```

## Repository Layout

```text
strawberry_preprocess/
├── scripts/
│   ├── convert_aihub_strawberry_to_yolo.py
│   ├── build_yolo_label_sanity_report.py
│   ├── train_yolo26m_strawberry.py
│   ├── eval_yolo26m_strawberry.py
│   ├── yolo26m_infer_to_crops.py
│   ├── build_yolo_ablation_plan.py
│   ├── build_yolo_trait_table.py
│   ├── merge_trait_predictions.py
│   ├── build_gdd_features.py
│   ├── build_kgcv_fruit_sequences.py
│   ├── predict_yolo_kgcv_yield.py
│   └── build_yolo_github_package.py
├── .github/workflows/
│   └── yolo_pipeline_tests.yml
├── PUBLISH_TO_GITHUB.md
├── DATA_AND_MODEL_LICENSES.md
├── reports/yolocsv/
│   ├── here.md
│   └── local_run_commands.md
├── tests/
│   ├── test_convert_aihub_strawberry_to_yolo.py
│   ├── test_yolo_local_scripts.py
│   └── test_yolo_growth_pipeline_tables.py
└── requirements.txt
```

## Input JSON Format

The converter expects JSON files shaped like this:

```json
{
  "images": {
    "image_id": 32435,
    "farm_id": "AI22_002",
    "crops_id": "BERRY_13_001",
    "fname": "딸기_설향_정상_13_001_220909091738.jpg",
    "width": "1071",
    "height": "1064",
    "date_captured": "2022-09-09 09:17:38",
    "growth_stage": "정식기"
  },
  "environments": [
    {"receive_date": "2022-09-01 17:26:05", "ti_value": "20.0"}
  ],
  "growth_index": {
    "crops_id": "BERRY_13_001",
    "measured_date": "2022-09-13",
    "leaf_cnt": "6"
  },
  "annotations": [
    {"bbox": [209.4, 495.72, 201.25, 114.2], "category_id": 0}
  ],
  "categories": [
    {"id": 0, "name": "잎"}
  ]
}
```

Important behavior:

- YOLO class is mapped by `categories[].name`, not by raw `category_id`.
- Default class names are `화방,과실,줄기,잎`.
- `bbox` is interpreted as COCO-style `[x, y, width, height]`.
- Environment values such as `ti_value` are summarized into image-level mean/min/max/latest columns.
- Growth values are exported with the `growth_` prefix.

## Setup

```bash
git clone <your-repo-url>
cd strawberry_yolo_kgcv_github
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For YOLO26m training, make sure the installed `ultralytics` version supports `yolo26m.pt`.

## Step 1. Convert JSON Labels to YOLO

Run a smoke test first.

```bash
python scripts/convert_aihub_strawberry_to_yolo.py \
  /path/to/label_json_dir \
  --image-root /path/to/image_root \
  --output-dir processed/yolo/aihub_strawberry_smoke \
  --class-names 화방,과실,줄기,잎 \
  --split 0.8,0.1,0.1 \
  --image-policy symlink \
  --max-files 50
```

Outputs:

```text
processed/yolo/aihub_strawberry_smoke/
├── aihub_strawberry_yolo.yaml
├── images/train|val|test/
├── labels/train|val|test/
└── metadata/
    ├── annotation_manifest.csv
    ├── image_growth_features.csv
    ├── conversion_errors.csv
    ├── class_distribution.csv
    └── conversion_summary.md
```

## Step 2. Visual Label Sanity Check

```bash
python scripts/build_yolo_label_sanity_report.py \
  --dataset-yaml processed/yolo/aihub_strawberry_smoke/aihub_strawberry_yolo.yaml \
  --output-dir reports/yolocsv/label_sanity_examples \
  --splits train,val,test \
  --sample-count 20
```

Open the generated images and verify that boxes and classes match the strawberry objects.

## Step 3. Train YOLO26m

Dry-run first:

```bash
python scripts/train_yolo26m_strawberry.py \
  --data processed/yolo/aihub_strawberry_smoke/aihub_strawberry_yolo.yaml \
  --model yolo26m.pt \
  --imgsz 1280 \
  --epochs 120 \
  --batch -1 \
  --device 0 \
  --project runs/yolo26m \
  --name strawberry_stage_det \
  --dry-run
```

If the config is correct, remove `--dry-run`.

## Step 4. Evaluate the Detector

```bash
python scripts/eval_yolo26m_strawberry.py \
  --data processed/yolo/aihub_strawberry_smoke/aihub_strawberry_yolo.yaml \
  --model runs/yolo26m/strawberry_stage_det/weights/best.pt \
  --imgsz 1280 \
  --batch 8 \
  --device 0 \
  --project runs/yolo26m \
  --name strawberry_stage_det_val
```

## Step 5. Export YOLO Crops

```bash
python scripts/yolo26m_infer_to_crops.py \
  --model runs/yolo26m/strawberry_stage_det/weights/best.pt \
  --source /path/to/images_or_video \
  --output-dir reports/yolocsv/yolo_crops \
  --imgsz 1280 \
  --conf 0.20 \
  --margin 0.30 \
  --output-size 256
```

Outputs:

```text
reports/yolocsv/yolo_crops/
├── crops/
└── detections_with_crops.csv
```

## Step 6. Build YOLO Ablation Plan

Generate a CSV with train/eval/crop commands for model size, image size, and confidence threshold comparisons.

```bash
python scripts/build_yolo_ablation_plan.py \
  --data processed/yolo/aihub_strawberry_smoke/aihub_strawberry_yolo.yaml \
  --output reports/yolocsv/yolo_ablation_plan.csv \
  --models yolo26n.pt,yolo26s.pt,yolo26m.pt \
  --imgsz 640,960,1280 \
  --conf 0.10,0.20,0.30 \
  --epochs 120 \
  --batch -1 \
  --device 0
```

## Step 7. Build Trait Table

This creates object-level features for downstream KGCV-style trait/growth modeling.

```bash
python scripts/build_yolo_trait_table.py \
  --annotation-manifest processed/yolo/aihub_strawberry_smoke/metadata/annotation_manifest.csv \
  --image-growth-features processed/yolo/aihub_strawberry_smoke/metadata/image_growth_features.csv \
  --output processed/yolo/trait_table.csv
```

Key columns:

- `fruit_id`
- `category_name`
- `bbox_area_px`
- `bbox_area_ratio`
- `bbox_aspect_ratio`
- `growth_*`
- `env_*`

## Step 8. Merge KGCV or External Trait Predictions

If a KGCV CNN or another crop trait model produces `diameter_mm`, `length_mm`, and `decimal_stage`, merge it into the trait table before sequence building.

Expected prediction CSV:

```text
fruit_id,diameter_mm,length_mm,decimal_stage,trait_model
C1:1:0,18.5,22.0,5.2,kgcv_cnn
```

Command:

```bash
python scripts/merge_trait_predictions.py \
  --trait-table processed/yolo/trait_table.csv \
  --trait-predictions /path/to/trait_predictions.csv \
  --output processed/yolo/trait_table_with_predictions.csv
```

If you do not have KGCV trait predictions yet, use `processed/yolo/trait_table.csv` directly in the next step.

## Step 9. Build GDD Features

```bash
python scripts/build_gdd_features.py \
  --image-growth-features processed/yolo/aihub_strawberry_smoke/metadata/image_growth_features.csv \
  --output processed/yolo/gdd_daily.csv \
  --base-temperature 10.0
```

Default temperature priority:

1. `env_ti_value_mean`
2. `env_tl_value_mean`
3. `env_cl_value_mean`

## Step 10. Build Fruit Sequences

```bash
python scripts/build_kgcv_fruit_sequences.py \
  --trait-table processed/yolo/trait_table_with_predictions.csv \
  --gdd processed/yolo/gdd_daily.csv \
  --output processed/yolo/fruit_sequences.csv
```

This adds:

- `date`
- `stage_score`
- `sequence_index`
- `delta_gdd`
- merged `gdd_day`, `gdd_cum`

If no trait prediction file exists, replace `trait_table_with_predictions.csv` with `trait_table.csv`.

## Step 11. Yield Baseline

```bash
python scripts/predict_yolo_kgcv_yield.py \
  --sequence processed/yolo/fruit_sequences.csv \
  --fruit-output processed/yolo/fruit_yield_predictions.csv \
  --group-output processed/yolo/group_yield_predictions.csv
```

This is a reproducible baseline, not the final scientific model. It uses measured `growth_fr_weight` when available; otherwise it falls back to trait/bbox proxy features.

## Roadmap Alignment

Implemented:

- AIHub JSON to YOLO conversion
- bbox/class sanity visualization
- YOLO26m train/eval wrappers
- YOLO inference crop export
- YOLO ablation command planner
- object trait table
- KGCV/external trait prediction merge adapter
- GDD table
- fruit sequence table
- yield baseline
- tests for conversion and downstream CSV contracts
- clean GitHub release package builder
- GitHub Actions test workflow

Next research extensions:

- Implement a direct KGCV pretrained CNN runner that writes the prediction CSV consumed by `merge_trait_predictions.py`.
- Replace the baseline yield proxy with KGCV `parameterNet` and `growthNet`.
- Add tracking/Hungarian matching for repeated imaging of the same fruit.
- Add ablation scripts for `imgsz`, model size, confidence threshold, crop margin, and GDD vs calendar days.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 pytest \
  tests/test_convert_aihub_strawberry_to_yolo.py \
  tests/test_yolo_local_scripts.py \
  tests/test_yolo_growth_pipeline_tables.py \
  tests/test_build_yolo_github_package.py \
  tests/test_build_yolo_ablation_plan.py \
  -q
```

## Build a Clean GitHub Package

If you are working from a larger local workspace that contains raw data, processed outputs, reports, and virtual environments, do not publish the whole workspace. Build a clean package instead:

```bash
python scripts/build_yolo_github_package.py \
  --source-root . \
  --output-root ../strawberry_yolo_kgcv_github
```

Then inspect:

```bash
cd ../strawberry_yolo_kgcv_github
find . -maxdepth 3 -type f | sort
PYTHONDONTWRITEBYTECODE=1 pytest \
  tests/test_convert_aihub_strawberry_to_yolo.py \
  tests/test_yolo_local_scripts.py \
  tests/test_yolo_growth_pipeline_tables.py \
  tests/test_build_yolo_github_package.py \
  tests/test_build_yolo_ablation_plan.py \
  -q
```

To publish the clean package, follow `PUBLISH_TO_GITHUB.md`.

## Large Data Safety

Do not run full raw/processed/report-wide preprocessing by default. Start with `--max-files 50`, inspect the class distribution and visual sanity outputs, then expand deliberately.
