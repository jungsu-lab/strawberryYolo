# Local YOLO26m Pipeline Commands

이 문서는 AIHub형 딸기 JSON 라벨을 YOLO dataset으로 변환하고, YOLO26m 학습/평가/추론 crop 생성을 로컬에서 실행하기 위한 최소 명령 모음이다.

## 1. JSON -> YOLO 변환 smoke test

먼저 작은 폴더나 `--max-files`로 변환만 확인한다.

```bash
cd strawberry_yolo_kgcv_github

python scripts/convert_aihub_strawberry_to_yolo.py \
  /path/to/label_json_dir \
  --image-root /path/to/image_root \
  --output-dir processed/yolo/aihub_strawberry_smoke \
  --class-names 화방,과실,줄기,잎 \
  --split 0.8,0.1,0.1 \
  --image-policy symlink \
  --max-files 50
```

출력:

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

## 2. 라벨 시각 검수

```bash
python scripts/build_yolo_label_sanity_report.py \
  --dataset-yaml processed/yolo/aihub_strawberry_smoke/aihub_strawberry_yolo.yaml \
  --output-dir reports/yolocsv/label_sanity_examples \
  --splits train,val,test \
  --sample-count 20
```

`reports/yolocsv/label_sanity_examples/` 아래의 이미지에서 bbox와 class가 맞는지 먼저 확인한다.

## 3. YOLO26m 학습 dry-run

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

dry-run 출력 설정이 맞으면 `--dry-run`을 제거해 실제 학습한다.

## 4. 평가 dry-run

```bash
python scripts/eval_yolo26m_strawberry.py \
  --data processed/yolo/aihub_strawberry_smoke/aihub_strawberry_yolo.yaml \
  --model runs/yolo26m/strawberry_stage_det/weights/best.pt \
  --imgsz 1280 \
  --batch 8 \
  --device 0 \
  --project runs/yolo26m \
  --name strawberry_stage_det_val \
  --dry-run
```

## 5. 추론 crop 생성

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

출력:

```text
reports/yolocsv/yolo_crops/
├── crops/
└── detections_with_crops.csv
```

## 6. YOLO ablation plan 생성

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

## 7. Trait table 생성

```bash
python scripts/build_yolo_trait_table.py \
  --annotation-manifest processed/yolo/aihub_strawberry_smoke/metadata/annotation_manifest.csv \
  --image-growth-features processed/yolo/aihub_strawberry_smoke/metadata/image_growth_features.csv \
  --output processed/yolo/trait_table.csv
```

## 8. KGCV/external trait prediction 병합

KGCV CNN 또는 별도 crop trait 모델이 다음 CSV를 만들면 trait table에 병합한다.

```text
fruit_id,diameter_mm,length_mm,decimal_stage,trait_model
C1:1:0,18.5,22.0,5.2,kgcv_cnn
```

```bash
python scripts/merge_trait_predictions.py \
  --trait-table processed/yolo/trait_table.csv \
  --trait-predictions /path/to/trait_predictions.csv \
  --output processed/yolo/trait_table_with_predictions.csv
```

KGCV trait prediction이 아직 없으면 다음 sequence 단계에서 `trait_table.csv`를 그대로 사용한다.

## 9. GDD feature 생성

```bash
python scripts/build_gdd_features.py \
  --image-growth-features processed/yolo/aihub_strawberry_smoke/metadata/image_growth_features.csv \
  --output processed/yolo/gdd_daily.csv \
  --base-temperature 10.0
```

## 10. KGCV-style fruit sequence 생성

```bash
python scripts/build_kgcv_fruit_sequences.py \
  --trait-table processed/yolo/trait_table_with_predictions.csv \
  --gdd processed/yolo/gdd_daily.csv \
  --output processed/yolo/fruit_sequences.csv
```

KGCV trait prediction이 아직 없으면 `--trait-table processed/yolo/trait_table.csv`로 실행한다.

## 11. Yield baseline 생성

```bash
python scripts/predict_yolo_kgcv_yield.py \
  --sequence processed/yolo/fruit_sequences.csv \
  --fruit-output processed/yolo/fruit_yield_predictions.csv \
  --group-output processed/yolo/group_yield_predictions.csv
```

## 12. 현재 구현 범위

구현 완료:

- AIHub형 JSON 라벨 -> YOLO dataset 변환
- 중복 `category_id`가 아니라 `categories.name` 기준 class 매핑
- bbox clipping 및 invalid bbox error 기록
- 이미지 단위 growth/environment feature CSV 생성
- 라벨 시각 검수 이미지 생성
- YOLO26m 학습/평가 dry-run wrapper
- YOLO inference 결과 crop 생성 wrapper
- YOLO ablation command plan 생성
- object-level trait table 생성
- KGCV/external trait prediction 병합
- GDD table 생성
- KGCV-style fruit sequence 생성
- yield baseline 생성

다음 구현 후보:

- KGCV pretrained CNN trait estimator 연결
- KGCV parameterNet/growthNet adapter
- tracking/Hungarian matching
- ablation runner
