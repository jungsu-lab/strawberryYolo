# GitHub Release Checklist

이 체크리스트는 현재 로컬 작업물을 GitHub 저장소로 공개하기 전에 확인할 항목이다.

## 1. 공개 범위 확인

- [ ] 원본 이미지, 원본 JSON, 압축 데이터셋은 GitHub에 올리지 않는다.
- [ ] `raw/`, `processed/`, `runs/`, model weight 파일은 `.gitignore`로 제외한다.
- [ ] 공개 가능한 코드, 테스트, 문서만 commit한다.
- [ ] 데이터 출처와 라이선스 조건을 `DATA_AND_MODEL_LICENSES.md`에 명확히 적는다.
- [ ] 프로젝트 code license는 `LICENSE`에 포함되어 있다. AIHub 데이터, KGCV, Ultralytics YOLO는 별도 조건을 확인한다.

## 2. 최소 실행 검증

```bash
cd strawberry_yolo_kgcv_github
PYTHONDONTWRITEBYTECODE=1 pytest \
  tests/test_convert_aihub_strawberry_to_yolo.py \
  tests/test_yolo_local_scripts.py \
  tests/test_yolo_growth_pipeline_tables.py \
  tests/test_build_yolo_github_package.py \
  tests/test_build_yolo_ablation_plan.py \
  -q
```

기대 결과:

```text
13 passed
```

## 3. Smoke Test 순서

작은 JSON 50개 이하로 먼저 돌린다.

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

그 다음:

```bash
python scripts/build_yolo_label_sanity_report.py \
  --dataset-yaml processed/yolo/aihub_strawberry_smoke/aihub_strawberry_yolo.yaml \
  --output-dir reports/yolocsv/label_sanity_examples \
  --sample-count 20
```

이미지 bbox가 맞는지 눈으로 확인한 뒤 후속 단계를 실행한다.

## 4. Full Pipeline Command Order

```bash
python scripts/build_yolo_trait_table.py \
  --annotation-manifest processed/yolo/aihub_strawberry_smoke/metadata/annotation_manifest.csv \
  --image-growth-features processed/yolo/aihub_strawberry_smoke/metadata/image_growth_features.csv \
  --output processed/yolo/trait_table.csv

# Optional: run this only after a KGCV/external crop trait model writes trait_predictions.csv.
python scripts/merge_trait_predictions.py \
  --trait-table processed/yolo/trait_table.csv \
  --trait-predictions /path/to/trait_predictions.csv \
  --output processed/yolo/trait_table_with_predictions.csv

python scripts/build_gdd_features.py \
  --image-growth-features processed/yolo/aihub_strawberry_smoke/metadata/image_growth_features.csv \
  --output processed/yolo/gdd_daily.csv \
  --base-temperature 10.0

python scripts/build_kgcv_fruit_sequences.py \
  --trait-table processed/yolo/trait_table_with_predictions.csv \
  --gdd processed/yolo/gdd_daily.csv \
  --output processed/yolo/fruit_sequences.csv

python scripts/predict_yolo_kgcv_yield.py \
  --sequence processed/yolo/fruit_sequences.csv \
  --fruit-output processed/yolo/fruit_yield_predictions.csv \
  --group-output processed/yolo/group_yield_predictions.csv
```

KGCV/external trait prediction이 아직 없으면 `build_kgcv_fruit_sequences.py`의 입력을 `processed/yolo/trait_table.csv`로 바꾼다.

실험 비교 계획은 별도로 생성한다.

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

## 5. YOLO Train/Eval

설정 확인:

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

실제 학습은 `--dry-run`을 제거한다.

## 6. Clean GitHub Package 생성

전체 작업공간에는 raw/processed/report 산출물이 많으므로 그대로 push하지 않는다. 공개용 패키지를 별도로 만든다.

```bash
python scripts/build_yolo_github_package.py \
  --source-root . \
  --output-root ../strawberry_yolo_kgcv_github
```

생성 후 확인:

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

GitHub 업로드 명령은 생성된 패키지의 `PUBLISH_TO_GITHUB.md`를 따른다.

## 7. README 점검

- [ ] 데이터 경로 예시가 로컬 절대경로로 고정되어 있지 않은지 확인한다.
- [ ] YOLO26m weight 다운로드 또는 Ultralytics 설치 조건을 명시한다.
- [ ] KGCV pretrained CNN/growthNet은 아직 연결 전이라는 점을 명시한다.
- [ ] 현재 yield는 baseline proxy이며 최종 연구 모델이 아니라는 점을 명시한다.

## 8. 다음 개발 이슈로 남길 항목

- [ ] KGCV pretrained CNN을 crop image에 연결해 `diameter_mm`, `length_mm`, `decimal_stage` prediction CSV 생성
- [ ] KGCV `parameterNet` / `growthNet` adapter 구현
- [ ] 동일 과실 추적용 Hungarian matching 구현
- [ ] YOLO `imgsz`, model size, confidence threshold, crop margin ablation runner 구현
- [ ] 실험 결과 표와 논문용 figure 자동 생성
