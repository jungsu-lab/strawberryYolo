# YOLO26m + KGCV Strawberry Growth Prediction Roadmap

작성일: 2026-07-03  
대상: YOLO26m detector와 `KGCV_Strawberry`를 결합한 온실 딸기 생장 궤적 및 수량 예측 파이프라인  
핵심 목표: `YOLO26m -> trait estimation -> GDD 기반 growth trajectory -> maturity/yield prediction`

---

## 1. 핵심 결론

이 프로젝트는 KGCV_Strawberry의 Faster R-CNN detector를 YOLO26m으로 교체하고, KGCV의 CNN trait estimator, parameterNet, growthNet, yield prediction 구조는 최대한 유지하면서 확장한다.

```text
기존 KGCV:
Faster R-CNN -> crop CNN -> parameterNet -> growthNet -> yield / maturity prediction

제안 구조:
YOLO26m -> crop CNN 또는 YOLO-crop fine-tuned trait estimator
        -> parameterNet -> growthNet -> maturity / yield prediction
```

YOLO26m은 최종 수량 예측 모델이 아니라 생장 예측에 필요한 객체 단위 feature를 뽑는 front-end이다. KGCV는 그 feature를 이용해 과실 크기, 생육단계, GDD, 성장곡선, 수확시점, 수량을 연결하는 knowledge-guided back-end로 둔다.

---

## 2. 검증된 기반 자료

### 2.1 KGCV_Strawberry

공식 저장소: <https://github.com/DigitalAgUMN/KGCV_Strawberry>

KGCV_Strawberry는 온실 딸기 과실 탐지와 생장 모델링을 위한 knowledge-guided computer vision framework이다. 저장소 기준 주요 파일은 다음과 같다.

| 구성 | 파일 | 역할 |
|---|---|---|
| Detection network | `KGCV_Strawberry_Train_FasterRCNN.py` | 딸기 bbox 및 main phenological stage detection |
| Trait estimation network | `KGCV_Strawberry_Train_CNN.py` | crop image에서 fruit size와 decimal phenological stage 추정 |
| Parameter network | `KGCV_Strawberry_Train_MLP-1_parameterNet.py` | sparse diameter/length sequence에서 S-shape growth curve parameter 추정 |
| Growth network | `KGCV_Strawberry_Train_MLP-1_growthNet.py` | `s_t`, `Delta GDD`, curve parameter로 `s_{t+Delta t}` 예측 |
| Integrated demo | `KGCV_Strawberry_demo.py` | 통합 데모 |

KGCV의 핵심 흐름:

```text
S-shape growth function:
f(x) = (y0 - s) / (1 + exp(-r(x - t0))) + s

parameterNet:
sparse 1 x 70 diameter/length sequence -> y0, s, r

growthNet:
s_t, Delta GDD, Para -> s_{t + Delta t}
```

라이선스: KGCV GitHub 저장소는 MIT license로 표시되어 있다.

### 2.2 KGCV Zenodo Dataset

데이터 DOI: <https://doi.org/10.5281/zenodo.10957909>

주요 파일:

| 파일 | 역할 | 우선순위 |
|---|---|---|
| `measurements.zip` | treatment-level, fruit-level ground truth, weather data | 매우 높음 |
| `models.zip` | KGCV pre-trained models | 높음 |
| `strawberry_img_random.zip` | object and phenological stage detection용 이미지 및 JSON label | 매우 높음 |
| `strawberry_img_tagged.zip` | fruit size와 decimal phenological stage label 포함 이미지 | 매우 높음 |
| `datasetProcessing.py` | image data merge/split 참고 스크립트 | 중간 |

`measurements.zip`에서 필요한 항목:

```text
Treatment-level:
- data_dryMatter_2022.csv
- data_dryMatter_2023.csv
- data_freshMatter_2022.csv
- data_freshMatter_2023.csv
- data_fruitNumber_2022.csv
- data_fruitNumber_2023.csv
- data_plantBiomass_2022.csv
- data_plantBiomass_2023.csv

Fruit-level:
- data_size_freshWeight_condition_2022_0N.csv
- data_size_freshWeight_condition_2022_50N.csv
- data_size_freshWeight_condition_2022_100N.csv
- data_size_freshWeight_condition_2022_150N.csv
- data_taggedFruit_diameter_2022.csv
- data_taggedFruit_diameter_2023.csv
- data_taggedFruit_length_2022.csv
- data_taggedFruit_length_2023.csv
- data_taggedFruit_freshMatter_2023.csv
- data_taggedFruit_lifespan_2023.csv

Weather:
- weather_daily_2022.csv
- weather_daily_2023.csv
```

`strawberry_img_tagged.zip` label 예시:

```json
{
  "label": "small g, 8.84, 7.62, 0.4"
}
```

의미:

- main stage: `small g`
- diameter: `8.84 mm`
- length: `7.62 mm`
- decimal phenological stage: `0.4`, 즉 DS-4 계열

Zenodo dataset license는 CC-BY 4.0이다.

### 2.3 YOLO26m

공식 문서: <https://docs.ultralytics.com/models/yolo26/>

YOLO26은 Ultralytics의 real-time vision model family이고 detection, instance segmentation, semantic segmentation, pose, classification, OBB task를 지원한다.

Detection checkpoint:

```text
yolo26n.pt
yolo26s.pt
yolo26m.pt
yolo26l.pt
yolo26x.pt
```

YOLO26m detection 공식 COCO 기준 메모:

| 모델 | 입력 | mAP50-95 | e2e mAP50-95 | CPU ONNX | T4 TensorRT | Params | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26m | 640 | 53.1 | 52.5 | 220.0 ms | 4.7 ms | 20.4M | 68.2B |

YOLO26 주요 특징:

- native end-to-end NMS-free inference
- DFL-free regression
- Progressive Loss + STAL, small object label coverage 개선
- MuSGD optimizer
- task-specific heads for detection, segmentation, pose, OBB

라이선스 주의:

- Ultralytics YOLO26 code/model/docs는 AGPL-3.0 또는 Enterprise license 체계로 안내된다.
- 학교 연구/수업/논문 프로토타입과 상용/웹서비스 배포는 라이선스 판단이 다르다.
- 공개 배포를 생각하면 초반에 KGCV MIT, Zenodo CC-BY 4.0, Ultralytics AGPL/Enterprise를 별도 문서로 정리한다.

---

## 3. 2025년 이후 YOLO/딸기 논문 후보 메모

아래 표는 이전 조사에서 나온 후보를 보존한 것이다. 이 로드맵에서는 구현 방향을 잡기 위한 메모로만 사용하고, 실제 인용 전에는 DOI, 논문 페이지, GitHub, weight 공개 여부를 재검증한다.

| 논문/모델명 | 핵심 내용 | 공개 여부 메모 | 활용 포인트 |
|---|---|---|---|
| `Evaluating YOLO Variants for Real-Time Multi-Object Detection of Strawberry Quality and Ripeness` | YOLOv7, YOLOv8, YOLOv11 딸기 품질/숙도 탐지 비교 | 이전 조사상 GitHub와 일부 결과/weight 공개 후보. 구현 전 재검증 필요 | YOLO fine-tuning baseline 참고 |
| `KGCV_Strawberry / Predicting the Growth Trajectory and Yield of Greenhouse Strawberries Based on Knowledge-Guided Computer Vision` | Faster R-CNN, CNN trait estimator, parameterNet, growthNet으로 growth/yield 예측 | GitHub 코드, Zenodo 데이터, pre-trained model 공개 | YOLO26m detector를 붙일 주 기반 |
| `WCS-YOLOv8s` | bud, flower, under-ripening fruit, ripening fruit 등 growth process stage detection | raw data 요청 가능, 코드/weight 공개 여부 재확인 필요 | 생육단계 class 설계 참고 |
| `RLK-YOLOv8` | greenhouse strawberry full growth cycle multi-stage detection | 공개 repo/weight 재확인 필요 | stage detection benchmark 참고 |
| `Real-Time Strawberry Ripeness Classification and Counting: Optimized YOLOv8s + Class-Aware MOT` | 숙도 분류, counting, tracking 구조 | 데이터 요청형 가능성, 코드/weight 재확인 필요 | YOLO -> tracking -> count/yield 연결 구조 참고 |
| `YOLOv11-GSF` | strawberry ripeness detection, GhostConv 등 구조 개선 | weight 공개 여부 재확인 필요 | ripeness detection architecture 참고 |
| `LBS-YOLO` | YOLOv11n 기반 lightweight strawberry ripeness detection | public dataset 사용 가능성, weight 공개 여부 재확인 필요 | 경량 모델 baseline |
| `YOLOv11-HRS` | hybrid channel-space attention 기반 ripeness detection | 공개 여부 재확인 필요 | attention 구조 참고 |
| `Ripe-Detection` | lightweight strawberry ripeness detection | 데이터 요청형 가능성 | edge/real-time detector 참고 |
| `STRAW-YOLO` | YOLOv8-pose 기반 fruit target/keypoint detection | repo 재확인 필요 | 수확 로봇, picking point 확장 참고 |
| `MFD-YOLO` | strawberry growth state detection 지향 | 공개 repo 재확인 필요 | growth state detection 직접 후보 |

로드맵상 문헌 활용 방식:

```text
1. KGCV는 생장 예측 구조의 주 기반으로 사용한다.
2. YOLO26m은 detector 교체와 small-object detection 개선 가설의 주 모델로 사용한다.
3. 2025년 이후 strawberry YOLO 논문들은 class 설계, ablation, 비교표, related work 근거로 사용한다.
4. 공개 weight가 없는 논문은 구현 baseline이 아니라 참고 문헌으로만 둔다.
```

---

## 4. 연구 질문과 논문화 포인트

주요 연구 질문:

```text
기존 KGCV의 Faster R-CNN detector를 YOLO26m으로 대체하면,
딸기 생육단계 탐지와 downstream growth/yield prediction이 얼마나 개선되는가?
```

세부 질문:

1. YOLO26m은 KGCV Faster R-CNN 대비 딸기 stage detection에서 더 높은 mAP/recall을 보이는가?
2. 작은 flower, small green fruit 같은 small object에서 YOLO26의 STAL 계열 장점이 실제로 도움이 되는가?
3. YOLO bbox crop을 KGCV CNN trait estimator에 넣으면 GT bbox crop 대비 trait error가 얼마나 증가하는가?
4. YOLO crop 기반으로 trait estimator를 fine-tuning하면 downstream growth/yield error가 줄어드는가?
5. YOLO class, decimal stage, GDD, fruit size feature 중 어떤 요소가 maturity/yield prediction에 가장 중요하게 작용하는가?

논문화 포인트:

```text
단순한 딸기 탐지가 아니라,
YOLO26m fine-tuned detector를 knowledge-guided growth simulator에 연결해
instant detection을 growth trajectory prediction으로 확장한다.
```

---

## 5. 추천 전체 아키텍처

```text
Input greenhouse strawberry image
↓
YOLO26m detector
  classes:
    flower
    small_g
    green
    white
    turning_red
    red
    overripe
↓
Detected bbox crop
↓
KGCV CNN trait estimator
  output:
    diameter_mm
    length_mm
    decimal_stage
↓
Sequence builder
  fruit_id
  date
  GDD
  diameter_t
  length_t
  stage_t
↓
KGCV parameterNet
  sparse diameter / length sequence -> growth curve parameters
↓
KGCV growthNet
  s_t, Delta GDD, parameters -> s_t+Delta_t
↓
Yield model
  predicted size + maturity date + fresh weight regression
↓
Output
  expected harvest date
  fruit-level future size
  plant/plot/bed-level yield
```

모듈 경계:

| 모듈 | 입력 | 출력 | 책임 |
|---|---|---|---|
| Label converter | KGCV LabelMe JSON, image files | YOLO txt labels, dataset yaml | detection dataset 생성 |
| YOLO trainer | YOLO dataset yaml | `best.pt`, metrics | YOLO26m fine-tuning |
| Detector evaluator | trained YOLO, test set | mAP, class AP, confusion matrix | detector 비교 |
| YOLO-to-crop adapter | image, YOLO bbox | crop jpg, `detections.csv` | KGCV CNN 입력 생성 |
| Trait estimator | crop image | diameter, length, decimal stage | 과실 trait 추정 |
| Sequence builder | detections, traits, date/weather | fruit sequence table | fruit-level time series 생성 |
| GDD processor | weather csv | daily/cumulative GDD | 생장 시간축 생성 |
| Growth model adapter | fruit sequence, KGCV MLP | future size, maturity date | 생장 궤적 예측 |
| Yield estimator | predicted fruit traits | fresh weight, plant/plot yield | 수량 예측 |
| Error analysis | all predictions | metrics, plots | 오차 분해 |

---

## 6. 추천 폴더 구조

독립 프로젝트로 만들 경우:

```text
strawberry_yolo_kgcv/
├── data/
│   ├── raw/
│   │   ├── strawberry_img_random/
│   │   ├── strawberry_img_tagged/
│   │   ├── measurements/
│   │   └── models/
│   ├── yolo_detection/
│   │   ├── images/train/
│   │   ├── images/val/
│   │   ├── images/test/
│   │   ├── labels/train/
│   │   ├── labels/val/
│   │   ├── labels/test/
│   │   └── strawberry_yolo.yaml
│   ├── crops_trait/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── processed/
│       ├── detections.csv
│       ├── traits.csv
│       ├── fruit_sequences.csv
│       ├── gdd_daily.csv
│       └── yield_targets.csv
├── kgcv_original/KGCV_Strawberry/
├── src/
│   ├── 00_download_check.py
│   ├── 01_convert_labelme_to_yolo.py
│   ├── 02_train_yolo26m.py
│   ├── 03_eval_yolo.py
│   ├── 04_yolo_infer_to_crops.py
│   ├── 05_trait_estimation.py
│   ├── 06_build_fruit_sequences.py
│   ├── 07_train_growth_model.py
│   ├── 08_predict_yield.py
│   └── 09_error_analysis.py
├── configs/
│   ├── strawberry_yolo.yaml
│   ├── yolo26m_train.yaml
│   └── kgcv_paths.yaml
├── runs/
│   ├── yolo26m/
│   ├── kgcv_baseline/
│   └── ablations/
└── notebooks/
    ├── 01_label_sanity_check.ipynb
    ├── 02_detection_error_analysis.ipynb
    ├── 03_growth_curve_visualization.ipynb
    └── 04_yield_report.ipynb
```

현재 `strawberry_preprocess`에 통합할 경우:

```text
strawberry_preprocess/
├── config/
│   ├── yolo26m_strawberry.yaml
│   └── kgcv_yolo_paths.yaml
├── scripts/
│   ├── convert_kgcv_labelme_to_yolo.py
│   ├── build_yolo_label_sanity_report.py
│   ├── train_yolo26m_strawberry.py
│   ├── eval_yolo26m_strawberry.py
│   ├── yolo26m_infer_to_crops.py
│   ├── build_yolo_trait_table.py
│   ├── build_kgcv_fruit_sequences.py
│   ├── run_yolo_kgcv_growth_prediction.py
│   └── predict_yolo_kgcv_yield.py
├── reports/yolocsv/
│   ├── here.md
│   ├── label_sanity_report.md
│   ├── yolo_detection_metrics.csv
│   ├── trait_estimation_metrics.csv
│   ├── growth_prediction_metrics.csv
│   └── yield_prediction_metrics.csv
└── tests/
    ├── test_convert_kgcv_labelme_to_yolo.py
    ├── test_yolo_crop_adapter.py
    ├── test_build_kgcv_fruit_sequences.py
    ├── test_gdd_calculation.py
    └── test_yolo_kgcv_growth_prediction.py
```

---

## 7. Label schema

YOLO class는 KGCV의 main phenological stage와 맞춘다.

```yaml
names:
  0: flower
  1: small_g
  2: green
  3: white
  4: turning_red
  5: red
  6: overripe
```

LabelMe raw label mapping:

```python
LABEL_MAP = {
    "flower": 0,
    "small g": 1,
    "small_g": 1,
    "green": 2,
    "white": 3,
    "turning red": 4,
    "turning_red": 4,
    "red": 5,
    "overripe": 6,
}
```

주의:

- YOLO에는 background class를 넣지 않는다.
- KGCV 원본 Faster R-CNN은 background를 포함했을 가능성이 있으므로 class id를 그대로 복사하지 않는다.
- `small g`, `small_g`, `turning red`, `turning_red`를 하나로 통일한다.
- `"small g, 8.84, 7.62, 0.4"` 같은 tagged label은 detection class와 trait label을 분리해서 저장한다.

---

## 8. Phase 0: 안전장치와 데이터 범위 확인

목표:

- 전체 dataset 다운로드/전처리 전에 입력 범위와 용량을 확인한다.
- 이 저장소의 대용량 데이터 안전장치를 지킨다.

원칙:

```text
1. 전체 raw/processed/reports/data 대상 전처리를 바로 실행하지 않는다.
2. 먼저 작은 sample 또는 단일 파일로 재현한다.
3. 전체 실행은 사용자 명시 승인 후만 수행한다.
4. 전체 실행 전에는 dry-run으로 처리 대상 수와 용량을 확인한다.
5. 품질 리포트류는 기본적으로 max-tables/max-input-mb 제한을 둔다.
```

초기 산출물:

```text
reports/yolocsv/data_scope_check.md
reports/yolocsv/download_manifest.csv
reports/yolocsv/sample_file_inventory.csv
```

체크 항목:

| 항목 | 확인 내용 |
|---|---|
| 데이터 용량 | `strawberry_img_random.zip`, `strawberry_img_tagged.zip`가 수 GB이므로 전체 압축 해제 전 위치와 용량 확인 |
| 저장 위치 | `strawberry_preprocess/data/external/kgcv/` 또는 별도 대용량 path |
| subset 전략 | 먼저 20-50장만 변환 및 sanity visualization |
| GPU 환경 | CUDA, PyTorch, ultralytics 설치 여부 |
| 라이선스 | KGCV MIT, Zenodo CC-BY 4.0, Ultralytics AGPL/Enterprise |

---

## 9. Phase 1: KGCV baseline 재현

목표:

YOLO를 붙이기 전에 KGCV 원본 pipeline이 돌아가는지 확인하고 baseline metric을 저장한다.

할 일:

1. KGCV repository clone 또는 현재 작업 폴더에 vendoring
2. Zenodo `models.zip`, `measurements.zip` 확인
3. 가능하면 작은 image subset으로 demo 실행
4. `KGCV_Strawberry_demo.py` 실행 경로와 dependency 확인
5. Faster R-CNN detection 결과, CNN trait 결과, growthNet 결과를 CSV로 저장

산출물:

```text
runs/kgcv_baseline/
├── baseline_detections.csv
├── baseline_traits.csv
├── baseline_growth_predictions.csv
├── baseline_yield_predictions.csv
└── baseline_metrics.json
```

확인할 것:

| 체크포인트 | 확인 내용 |
|---|---|
| 이미지 로딩 | EXIF 회전 문제 여부 |
| bbox 좌표 | 원본 좌표계인지, resize 후 좌표계인지 |
| class 이름 | `small g` vs `small_g`, `turning red` vs `turning_red` |
| crop 크기 | KGCV CNN 입력이 256 x 256인지 |
| crop margin | 원본 KGCV crop 방식과 margin |
| 단위 | diameter/length가 mm인지 |
| normalization | trait estimator 입력/출력 normalization 방식 |
| GDD | weather csv 기준 재계산 가능 여부 |

성공 기준:

```text
1. KGCV demo가 최소 sample에서 실행된다.
2. detection -> trait -> growth 결과가 한 번 이상 끝까지 생성된다.
3. baseline output schema가 문서화된다.
```

---

## 10. Phase 2: LabelMe JSON -> YOLO format 변환

목표:

KGCV random image detection JSON을 YOLO detection 학습 포맷으로 변환한다.

YOLO dataset yaml:

```yaml
path: data/yolo_detection
train: images/train
val: images/val
test: images/test

names:
  0: flower
  1: small_g
  2: green
  3: white
  4: turning_red
  5: red
  6: overripe
```

변환 로직:

```python
def normalize_label(label: str) -> str:
    main = label.split(",")[0].strip().lower()
    main = main.replace("_", " ")
    return main


label_map = {
    "flower": 0,
    "small g": 1,
    "green": 2,
    "white": 3,
    "turning red": 4,
    "red": 5,
    "overripe": 6,
}
```

bbox 변환:

```python
x1 = min(p1_x, p2_x)
y1 = min(p1_y, p2_y)
x2 = max(p1_x, p2_x)
y2 = max(p1_y, p2_y)

x_center = ((x1 + x2) / 2) / image_width
y_center = ((y1 + y2) / 2) / image_height
w = (x2 - x1) / image_width
h = (y2 - y1) / image_height
```

YOLO txt line:

```text
class_id x_center y_center width height
```

함정:

- KGCV Faster R-CNN 코드가 이미지를 `scale=1/4`로 resize해서 학습했을 수 있다.
- YOLO label은 원본 이미지 크기 기준으로 정규화하는 것이 안전하다.
- 이미지 자체를 resize해서 저장한다면 label 좌표도 같은 resize 기준으로 맞춰야 한다.
- LabelMe polygon/rectangle 형식이 섞여 있으면 polygon의 min/max bbox로 변환하되, 이후 시각 검수한다.

sanity check:

```text
1. bbox가 과실/꽃을 정확히 감싸는가?
2. class color가 맞는가?
3. EXIF 회전 때문에 bbox가 틀어지지 않았는가?
4. 작은 flower/small_g가 사라지지 않았는가?
5. overripe class가 너무 적지 않은가?
6. train/val/test split에 중복 이미지가 없는가?
```

산출물:

```text
data/yolo_detection/
├── images/train/*.jpg
├── images/val/*.jpg
├── images/test/*.jpg
├── labels/train/*.txt
├── labels/val/*.txt
├── labels/test/*.txt
└── strawberry_yolo.yaml

reports/yolocsv/
├── yolo_label_class_distribution.csv
├── yolo_label_conversion_errors.csv
└── label_sanity_examples/
```

---

## 11. Phase 3: YOLO26m fine-tuning

목표:

YOLO26m을 KGCV strawberry stage detection dataset에 fine-tuning한다.

기본 학습 코드:

```python
from ultralytics import YOLO

model = YOLO("yolo26m.pt")

model.train(
    data="configs/strawberry_yolo.yaml",
    imgsz=1280,
    epochs=120,
    batch=-1,
    device=0,
    patience=30,
    project="runs/yolo26m",
    name="strawberry_stage_det",
)
```

추천 hyperparameter:

| 항목 | 추천값 | 이유 |
|---|---:|---|
| model | `yolo26m.pt` | 정확도와 속도 균형 |
| imgsz | 960 또는 1280 | flower/small_g 같은 작은 객체 대응 |
| epochs | 100-150 | class imbalance 가능성 |
| batch | `-1` 또는 GPU에 맞춤 | Ultralytics auto batch 활용 |
| patience | 30 | val 성능 plateau 감지 |
| conf eval | 0.10, 0.20, 0.30 sweep | 작은 객체 recall 확보 |
| max_det | 넉넉히 | 이미지당 객체 수가 많을 수 있음 |
| optimizer | YOLO26 default 우선 | 공식 recipe 유지 |

augmentation 시작값:

```yaml
fliplr: 0.5
flipud: 0.0
hsv_h: 0.01
hsv_s: 0.3
hsv_v: 0.2
degrees: 5
translate: 0.05
scale: 0.5
mosaic: 0.3
close_mosaic: 10
```

주의:

- 숙도/생육단계는 색상 정보가 중요하다.
- `hsv_s`, `hsv_v`를 너무 세게 주면 white, turning_red, red, overripe 경계가 흐려질 수 있다.
- 첫 실험은 augmentation을 약하게 시작하고 class-wise AP를 보면서 조절한다.

baseline 비교:

| ID | Detector | 입력 크기 | 목적 |
|---|---|---:|---|
| B0 | KGCV Faster R-CNN | KGCV 원본 | 원본 baseline |
| B1 | YOLO26m | 640 | 최소 비용 baseline |
| B2 | YOLO26m | 960 | 중간 입력 크기 |
| B3 | YOLO26m | 1280 | small-object recall 중점 |
| B4 | YOLO26s 또는 YOLO26n | 960/1280 | 경량화 baseline |
| B5 | YOLO26m + crop CNN downstream | 1280 | 최종 pipeline 전 단계 |

---

## 12. Phase 4: YOLO detector 평가

기본 metric:

- mAP50
- mAP50-95
- precision
- recall
- class-wise AP
- confusion matrix
- PR curve
- FPS / latency

class-wise error 포인트:

| Class | 자주 생길 문제 | 대응 |
|---|---|---|
| flower | 너무 작아서 miss | `imgsz=1280`, small-object recall 중심 평가 |
| small_g | leaf/background와 혼동 | bbox sanity check, class imbalance 확인 |
| green | 잎과 색이 비슷함 | crop 중심 augmentation, hard negative 확인 |
| white | 빛 반사와 혼동 | exposure augmentation 조절 |
| turning_red | red/white 중간이라 헷갈림 | decimal stage와 후처리 |
| red | 상대적으로 쉬움 | count/yield 핵심 class |
| overripe | 데이터 부족 가능성 | oversampling, class별 threshold 검토 |

평가 산출물:

```text
reports/yolocsv/
├── yolo_detection_metrics.csv
├── yolo_detection_class_ap.csv
├── yolo_detection_confusion_matrix.png
├── yolo_detection_pr_curve.png
├── yolo_detection_error_examples.md
└── yolo_detector_comparison.md
```

---

## 13. Phase 5: YOLO -> KGCV adapter

목표:

YOLO detection 결과를 KGCV trait estimator가 먹을 수 있는 crop과 metadata table로 변환한다.

입력:

```text
image_path
date
plant_id 또는 plot_id
weather/GDD 정보
trained_yolo26m.pt
KGCV trait CNN model
```

출력 CSV schema:

```text
image_id,date,plant_id,det_id,yolo_class,yolo_conf,
x1,y1,x2,y2,crop_path,
diameter_mm,length_mm,decimal_stage,gdd
```

adapter pseudo-code:

```python
from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO


CLASS_MAP = {
    0: "flower",
    1: "small g",
    2: "green",
    3: "white",
    4: "turning red",
    5: "red",
    6: "overripe",
}


def crop_with_margin(img: Image.Image, xyxy: list[float], margin: float = 0.30) -> Image.Image:
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    side = max(w, h) * (1 + margin)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    nx1 = max(0, int(cx - side / 2))
    ny1 = max(0, int(cy - side / 2))
    nx2 = min(img.width, int(cx + side / 2))
    ny2 = min(img.height, int(cy + side / 2))

    return img.crop((nx1, ny1, nx2, ny2)).resize((256, 256))


def infer_one_image(image_path: Path, yolo: YOLO, output_crop_dir: Path) -> pd.DataFrame:
    img = Image.open(image_path).convert("RGB")
    results = yolo.predict(str(image_path), imgsz=1280, conf=0.20, verbose=False)

    rows = []
    for i, box in enumerate(results[0].boxes):
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = box.xyxy[0].cpu().numpy().tolist()

        crop = crop_with_margin(img, xyxy, margin=0.30)
        crop_path = output_crop_dir / f"{image_path.stem}_{i:03d}.jpg"
        crop.save(crop_path)

        rows.append(
            {
                "image_path": str(image_path),
                "det_id": i,
                "class_id": cls_id,
                "class_name": CLASS_MAP[cls_id],
                "conf": conf,
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3],
                "crop_path": str(crop_path),
            }
        )

    return pd.DataFrame(rows)
```

crop margin ablation:

| margin | 목적 |
|---:|---|
| 0.1 | tight crop, bbox 정확할 때 |
| 0.2 | 중간 |
| 0.3 | KGCV 방식과 유사한 기본 후보 |
| 0.5 | bbox 흔들림에 강하지만 background 증가 |

추천 시작값은 `0.3`이다.

---

## 14. Phase 6: Trait estimator 연결

### 선택지 A: KGCV CNN 그대로 사용

가장 빠른 MVP 경로:

```text
YOLO26m bbox
↓
crop 256 x 256
↓
KGCV pretrained CNN
↓
diameter, length, decimal_stage
```

장점:

- 구현 빠름
- KGCV와 직접 비교 가능
- 연구 스토리 명확

단점:

- KGCV CNN은 GT bbox 기반 crop에 익숙할 수 있다.
- YOLO bbox 오차가 trait estimation error로 전파될 수 있다.

### 선택지 B: YOLO crop 기반으로 CNN fine-tuning

더 좋은 최종 경로:

```text
1. tagged image에서 GT bbox crop으로 CNN baseline 학습
2. 학습된 YOLO26m으로 tagged image inference
3. YOLO bbox crop 생성
4. YOLO crop + GT trait label로 CNN fine-tuning
5. GT crop CNN vs YOLO crop CNN 비교
```

이 방식은 실제 inference crop distribution에 trait estimator를 적응시킨다.

### 선택지 C: YOLO26m-seg 확장

과실 크기 예측이 bbox crop만으로 흔들리면 `yolo26m-seg.pt`로 확장한다.

단, 첫 버전에서는 detection YOLO26m + KGCV CNN이 더 현실적이다. segmentation은 mask label 변환, mask 품질, 평가 지표가 추가된다.

trait metric:

- diameter MAE / RMSE / R2
- length MAE / RMSE / R2
- decimal stage MAE / RMSE / R2
- stage class confusion
- crop margin별 error

---

## 15. Phase 7: Fruit identity / tracking

생장 예측은 같은 과실을 시간축으로 따라가야 한다.

### 15.1 tagged fruit 중심 데이터인 경우

KGCV tagged fruit 데이터의 fruit ID 또는 파일명 규칙을 활용해 sequence를 만든다.

```text
fruit_id,date,diameter_mm,length_mm,decimal_stage,gdd
F001,2023-04-01,8.84,7.62,0.4,120.5
F001,2023-04-03,10.10,8.40,0.7,135.2
F001,2023-04-05,13.20,10.50,1.2,150.8
```

### 15.2 greenhouse video인 경우

Ultralytics tracking mode를 사용한다.

```python
from ultralytics import YOLO

model = YOLO("runs/yolo26m/strawberry_stage_det/weights/best.pt")
results = model.track(source="greenhouse_video.mp4", imgsz=1280, persist=True)
```

tracker 후보:

- ByteTrack
- BoT-SORT

### 15.3 주기적 촬영 이미지인 경우

영상 tracking이 아니라 Hungarian matching이 현실적이다.

matching feature:

- bbox center distance
- bbox size similarity
- class transition plausibility
- diameter/length similarity
- same plant/row/camera constraint

허용 stage transition:

```text
flower -> small_g -> green -> white -> turning_red -> red -> overripe
```

허용:

- same stage
- next stage
- skip 1 stage

차단:

- red -> green
- overripe -> white
- large fruit -> flower

matching score 예시:

```text
score =
  0.35 * normalized_center_distance
+ 0.20 * bbox_size_difference
+ 0.25 * trait_size_difference
+ 0.20 * stage_transition_penalty
```

tracking/sequence metric:

- ID switch count
- fruit sequence completeness
- missed observation rate
- stage transition violation rate

---

## 16. Phase 8: GDD 계산

KGCV growthNet 입력에 `Delta GDD`가 들어가므로 weather data를 맞춘다.

기본 GDD 계산:

```python
T_avg = (T_max + T_min) / 2
GDD_day = max(T_avg - T_base, 0)
GDD_cum = GDD_day.cumsum()
```

출력 schema:

```text
date,tmax,tmin,tavg,gdd_day,gdd_cum
2023-04-01,24.1,15.2,19.65,14.65,120.5
```

주의:

- `T_base`는 KGCV 원본 코드/논문 기준과 맞춘다.
- 처음에는 KGCV 원본 전처리를 최대한 복제한다.
- 이후 ablation으로 `GDD` vs `calendar days`를 비교한다.

산출물:

```text
data/processed/gdd_daily.csv
reports/yolocsv/gdd_reproduction_check.md
```

---

## 17. Phase 9: Growth trajectory prediction

입력 sequence:

```text
fruit_id,date,gdd_cum,diameter_mm,length_mm,decimal_stage,yolo_class
F001,2023-04-01,120.5,8.84,7.62,0.4,small_g
F001,2023-04-03,135.2,10.10,8.40,0.7,small_g
F001,2023-04-05,150.8,13.20,10.50,1.2,green
```

parameterNet:

```text
input:
  sparse 1 x 70 diameter sequence
  sparse 1 x 70 length sequence

output:
  growth curve parameters
```

growthNet:

```text
input:
  current_size
  delta_GDD
  curve_parameters

output:
  future_size
```

maturity prediction 방식:

```text
방식 1:
decimal_stage >= red_threshold
또는
yolo_class == red

방식 2:
maturity_score =
  sigmoid(a * predicted_diameter
        + b * predicted_length
        + c * decimal_stage
        + d * yolo_stage_score)
```

추천은 방식 2이다. YOLO class만 쓰면 turning_red와 red 경계에서 흔들릴 수 있다.

growth metric:

- future diameter RMSE
- future length RMSE
- prediction horizon별 RMSE
- maturity date error
- growth curve parameter error

horizon 예시:

```text
+1 observation
+3 days
+7 days
+14 days
harvest maturity date
```

---

## 18. Phase 10: Yield prediction

fruit-level fresh weight 모델:

```text
fresh_weight_g = f(diameter_mm, length_mm, decimal_stage, condition)
```

추천 feature:

- diameter
- length
- diameter^2
- length^2
- diameter * length
- diameter^2 * length
- decimal_stage
- yolo_class_onehot
- condition, 사용 가능할 때만

모델 후보 순서:

1. Linear Regression
2. Random Forest
3. MLP

처음부터 복잡한 모델을 붙이지 않는다. 먼저 해석 가능한 baseline을 만들고, 이후 non-linear 모델로 확장한다.

plant/plot-level yield aggregation:

```text
predicted_yield_plant =
  sum(predicted_fresh_weight of fruits expected to reach red stage within harvest window)

Y_plant = sum_i FW_i * I(maturity_i within target window)

soft aggregation:
Y_plant = sum_i FW_i * P(mature_i)
```

yield metric:

- fruit-level fresh weight MAE
- plant-level yield RMSE
- plot/treatment-level yield RMSE
- MAPE
- R2

---

## 19. 실험 설계

핵심 실험표:

| ID | Detector | Trait estimator | Growth model | 목적 |
|---|---|---|---|---|
| E0 | KGCV Faster R-CNN | KGCV CNN | KGCV MLP | 원본 baseline |
| E1 | YOLO26m | KGCV CNN | KGCV MLP | detector 교체 효과 |
| E2 | YOLO26m | YOLO-crop fine-tuned CNN | KGCV MLP | crop domain adaptation 효과 |
| E3 | YOLO26m | bbox-size heuristic | KGCV MLP | CNN 없이 가능한지 |
| E4 | YOLO26m | CNN | no GDD, calendar day only | GDD 효과 |
| E5 | YOLO26s/YOLO26n | CNN | KGCV MLP | 경량화 비교 |
| E6 | YOLO26m-seg optional | mask-based trait | KGCV MLP | segmentation 확장 |

Detection ablation:

| 항목 | 후보 |
|---|---|
| imgsz | 640, 960, 1280 |
| model size | YOLO26n, YOLO26s, YOLO26m |
| class design | 7 classes vs fruit/flower binary |
| confidence threshold | 0.10, 0.20, 0.30 |
| crop margin | 0.1, 0.3, 0.5 |
| head mode | default end-to-end vs one-to-many head |

Growth ablation:

| 항목 | 비교 |
|---|---|
| time feature | GDD vs calendar day |
| stage source | YOLO class vs CNN decimal stage |
| observation count | single observation vs multi observation |
| crop source | GT bbox crop vs YOLO bbox crop |
| trait source | GT trait vs predicted trait |

오차 분해:

```text
1. Detector error
2. Trait estimator error
3. Tracking/identity error
4. Growth model error
5. Yield aggregation error
```

---

## 20. 데이터 split 전략

나쁜 split:

```text
같은 fruit_id의 4월 1일 이미지는 train
같은 fruit_id의 4월 3일 이미지는 test
```

이렇게 하면 growth prediction에서 leakage가 발생한다.

추천 split:

### Detection

- image-level random split 가능
- 단, 같은 촬영 날짜/식물/카메라 편향 확인
- class distribution을 train/val/test에서 비교

### Growth prediction

- fruit_id 기준 split
- year 기준 split
- treatment 기준 held-out split

강력 추천:

```text
Train: 2022
Test: 2023
```

또는:

```text
Train: 일부 treatment
Test: held-out treatment
```

cross-year evaluation을 넣으면 논문 설득력이 좋아진다.

---

## 21. 평가 지표

### 21.1 Detection

- mAP50
- mAP50-95
- class-wise precision
- class-wise recall
- confusion matrix
- FPS / latency

### 21.2 Trait estimation

- Diameter MAE / RMSE / R2
- Length MAE / RMSE / R2
- Decimal stage MAE / RMSE / R2

### 21.3 Tracking / sequence

- ID switch count
- fruit sequence completeness
- missed observation rate
- stage transition violation rate

### 21.4 Growth trajectory

- future diameter RMSE
- future length RMSE
- prediction horizon별 RMSE
- maturity date error

### 21.5 Yield

- fruit-level fresh weight MAE
- plant-level yield RMSE
- plot/treatment-level yield RMSE
- MAPE
- R2

---

## 22. MVP v1

처음 구현 목표는 전체 자동화가 아니라 end-to-end 흐름이 한 번 연결되는 것이다.

MVP v1 범위:

```text
1. KGCV random image JSON -> YOLO format 변환
2. YOLO26m detection fine-tuning
3. YOLO bbox crop 생성
4. KGCV pretrained CNN으로 diameter/length/stage 추정
5. tagged fruit 일부 sequence에 growthNet 적용
6. predicted growth curve visualization
```

MVP demo:

```text
Input image
↓
YOLO detection visualization
↓
detected fruit crop
↓
predicted diameter / length / stage
↓
future growth curve
```

MVP 산출물:

```text
reports/yolocsv/mvp_detection_examples/
reports/yolocsv/mvp_crop_examples/
reports/yolocsv/mvp_growth_curve_examples/
reports/yolocsv/mvp_summary.md
```

MVP 성공 기준:

```text
1. 샘플 이미지에서 YOLO detection이 나온다.
2. detection bbox crop이 256 x 256으로 저장된다.
3. crop에서 trait prediction이 나온다.
4. 하나 이상의 fruit sequence에 대해 future size curve가 그려진다.
```

---

## 23. 최종 구현 버전

최종 pipeline:

```text
Image folder or greenhouse video
↓
YOLO26m detection
↓
Object tracking / fruit matching
↓
Trait estimation
↓
GDD merge
↓
Fruit-level growth trajectory prediction
↓
Maturity date prediction
↓
Fresh weight regression
↓
Plant/plot-level yield forecast
↓
Dashboard / report
```

최종 산출물:

```text
1. yolo26m_strawberry_best.pt
2. yolo_detection_metrics.json
3. trait_estimation_metrics.json
4. growth_prediction_metrics.json
5. yield_prediction_metrics.json
6. ablation_table.csv
7. visualization figures
8. reproducible GitHub repo or local reproducible pipeline
9. paper/report draft
```

---

## 24. 바로 만들 핵심 파일

| 우선순위 | 파일 | 역할 |
|---:|---|---|
| 1 | `scripts/convert_kgcv_labelme_to_yolo.py` | KGCV LabelMe JSON을 YOLO txt로 변환, class mapping, split, error table 생성 |
| 2 | `scripts/build_yolo_label_sanity_report.py` | 변환 bbox를 이미지 위에 그려 sanity check png와 class distribution 생성 |
| 3 | `scripts/train_yolo26m_strawberry.py` | `yolo26m.pt` fine-tuning, run name/config 저장 |
| 4 | `scripts/eval_yolo26m_strawberry.py` | detector metric, confusion matrix, PR curve 생성 |
| 5 | `scripts/yolo26m_infer_to_crops.py` | trained YOLO inference, bbox crop, `detections.csv` 생성 |
| 6 | `scripts/build_yolo_trait_table.py` | crop image를 KGCV CNN에 입력해 diameter/length/decimal_stage 예측 |
| 7 | `scripts/build_kgcv_fruit_sequences.py` | fruit_id/date/GDD/trait 연결, sparse sequence 생성 |
| 8 | `scripts/run_yolo_kgcv_growth_prediction.py` | KGCV growthNet 연결, future size/maturity 예측 |
| 9 | `scripts/predict_yolo_kgcv_yield.py` | fresh weight regression, plant/plot/treatment yield aggregation |

---

## 25. 테스트 계획

초기 테스트 파일:

```text
tests/test_convert_kgcv_labelme_to_yolo.py
tests/test_yolo_crop_adapter.py
tests/test_build_kgcv_fruit_sequences.py
tests/test_gdd_calculation.py
tests/test_yolo_kgcv_growth_prediction.py
```

필수 테스트:

### Label conversion

```text
1. "small g, 8.84, 7.62, 0.4" -> class_id 1
2. "turning red" -> class_id 4
3. bbox 좌표가 0-1 범위로 정규화된다.
4. polygon/rectangle label이 bbox로 변환된다.
5. 알 수 없는 label은 error csv에 기록된다.
```

### Crop adapter

```text
1. bbox margin 0.3 적용 후 crop이 이미지 경계를 넘지 않는다.
2. crop output size가 256 x 256이다.
3. detection row와 crop path 수가 일치한다.
```

### GDD

```text
1. T_avg = (T_max + T_min) / 2
2. GDD_day = max(T_avg - T_base, 0)
3. GDD_cum이 날짜순 누적으로 계산된다.
```

### Sequence

```text
1. 같은 fruit_id 내 날짜순 정렬
2. red -> green 같은 비정상 stage transition flag 생성
3. sparse 1 x 70 sequence shape 검증
```

---

## 26. 리스크와 대응

| 리스크 | 증상 | 대응 |
|---|---|---|
| 작은 flower/small_g miss | flower recall 낮음 | `imgsz=1280`, small-object 중심 augmentation, YOLO26 P2 yaml 검토 |
| class imbalance | overripe AP 낮음 | oversampling, class별 error 분석, threshold sweep |
| YOLO bbox crop이 CNN에 안 맞음 | diameter/length 오차 증가 | YOLO crop으로 CNN fine-tuning |
| 같은 과실 추적 실패 | growth curve가 튐 | plant/camera/date constraint + Hungarian matching |
| 데이터 leakage | test 성능만 비정상적으로 높음 | fruit_id/year/treatment 기준 split |
| GDD 계산 불일치 | KGCV 재현 실패 | 원본 코드와 weather preprocessing 먼저 맞추기 |
| 라이선스 이슈 | 공개/상용화 애매함 | KGCV MIT, Zenodo CC-BY, Ultralytics AGPL/Enterprise 별도 정리 |
| 대용량 데이터 과부하 | WSL/VS Code hang, 디스크 사용량 증가 | subset 우선, dry-run, 명시 승인 전 전체 실행 금지 |

---

## 27. 논문/보고서 목차

```text
1. Introduction
   - greenhouse strawberry yield prediction 필요성
   - instant CV detection의 한계
   - knowledge-guided growth prediction 필요성

2. Related Work
   - strawberry detection
   - YOLO-based agricultural detection
   - crop growth modeling
   - KGCV-strawberry

3. Materials and Methods
   3.1 Dataset
   3.2 YOLO26m detector fine-tuning
   3.3 Trait estimation from YOLO crops
   3.4 Fruit sequence construction and GDD calculation
   3.5 Growth trajectory modeling
   3.6 Yield prediction
   3.7 Evaluation metrics

4. Experiments
   - Faster R-CNN vs YOLO26m
   - GT bbox vs YOLO bbox
   - GDD vs calendar day
   - image size ablation
   - crop margin ablation

5. Results
   - detection performance
   - trait estimation performance
   - growth prediction performance
   - yield prediction performance
   - error analysis

6. Discussion
   - YOLO detector의 장점
   - 작은 flower/green fruit miss 문제
   - tracking limitation
   - domain adaptation 필요성
   - license and reproducibility

7. Conclusion
```

---

## 28. 추천 작업 순서

### 1단계: 재현

- [ ] KGCV demo 실행
- [ ] Zenodo model/data 경로 정리
- [ ] 원본 baseline metric 저장

### 2단계: detection 변환

- [ ] LabelMe JSON 파싱
- [ ] YOLO txt 생성
- [ ] bbox sanity visualization
- [ ] class distribution 확인

### 3단계: YOLO26m 학습

- [ ] `imgsz=640` baseline
- [ ] `imgsz=960`
- [ ] `imgsz=1280`
- [ ] class-wise AP 비교
- [ ] `best.pt` 저장

### 4단계: KGCV 연결

- [ ] YOLO inference
- [ ] bbox crop
- [ ] KGCV CNN trait estimation
- [ ] `detections.csv` + `traits.csv` 생성

### 5단계: sequence/growth

- [ ] GDD 계산
- [ ] fruit sequence 생성
- [ ] parameterNet/growthNet 연결
- [ ] future size prediction
- [ ] maturity date prediction

### 6단계: yield

- [ ] fresh weight regression
- [ ] plant/plot/treatment yield aggregation
- [ ] yield metric 계산

### 7단계: 논문화

- [ ] baseline comparison
- [ ] ablation table
- [ ] error analysis
- [ ] figures
- [ ] report/paper 작성

---

## 29. 구현 전 최종 결정사항

코드를 짜기 전에 확정해야 할 것:

1. 데이터 저장 위치
   - `strawberry_preprocess/data/external/kgcv/`에 둘지
   - 별도 대용량 path에 둘지

2. 첫 학습 범위
   - 전체 random image를 쓸지
   - 50-200장 subset으로 먼저 변환/학습 smoke test를 할지

3. YOLO26 사용 방식
   - `yolo26m.pt` detection만 사용할지
   - small object를 위해 P2 yaml 추가 실험을 열어둘지

4. KGCV 모델 사용 방식
   - pre-trained CNN/MLP를 바로 쓸지
   - 먼저 KGCV 원본 학습을 재현할지

5. 최종 산출물 목표
   - 연구 보고서 중심
   - 실행 가능한 pipeline 중심
   - 데모/dashboard 중심

추천 기본 결정:

```text
1. 데이터는 먼저 subset만 사용한다.
2. detection은 yolo26m.pt로 시작한다.
3. KGCV CNN/MLP는 pre-trained model을 먼저 연결한다.
4. MVP가 돌아간 뒤 YOLO crop fine-tuning과 ablation으로 확장한다.
```

---

## 30. 다음 구현 시작점

가장 먼저 만들 파일:

```text
scripts/convert_kgcv_labelme_to_yolo.py
tests/test_convert_kgcv_labelme_to_yolo.py
```

첫 번째 기능의 성공 기준:

```text
1. KGCV LabelMe JSON 하나를 읽는다.
2. main stage label을 YOLO class id로 변환한다.
3. bbox를 YOLO normalized xywh로 변환한다.
4. YOLO txt label을 저장한다.
5. conversion error를 CSV로 저장한다.
6. 작은 sample directory에서 변환이 끝난다.
```

그 다음:

```text
scripts/build_yolo_label_sanity_report.py
```

성공 기준:

```text
1. 변환된 YOLO label을 원본 이미지 위에 다시 그린다.
2. class별 색상을 다르게 표시한다.
3. sample 20장에 대해 sanity image를 저장한다.
4. class distribution table을 만든다.
```

이 두 파일이 맞으면 이후 YOLO 학습, crop adapter, KGCV 연결이 안정적으로 이어진다.
