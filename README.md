# 지능형 객체 추적 시스템

YOLO11n으로 사람(및 고양이·개·새)을 검출하고, **직접 구현한 ByteTrack과 그 개선판**으로 프레임 간 같은 객체에 같은 ID를 유지하는
다중 객체 추적 시스템입니다. 추적 결과·성능 비교는 PySide6 GUI로 확인합니다.

* 과제 요구 사항 10개와 구현 대응표: [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md)
* ByteTrack 원리와 개선 내용: [`docs/BYTETRACK.md`](docs/BYTETRACK.md)

| 데이터셋 | 용도 | 결과 가중치 |
|---|---|---|
| [COCO](https://cocodataset.org/#download) | 사람·고양이·개·새 검출 | `models/coco.pt` |
| [MOT17](https://motchallenge.net/data/MOT17/) · [MOT20](https://motchallenge.net/data/MOT20/) | 보행자 추적 평가(HOTA·IDF1·MOTA·IDSW), 파라미터 튜닝, 검출기 학습 | `models/person.pt` (선택) |
| [CrowdHuman](https://www.crowdhuman.org/download.html) | 사람이 많이 겹치는 장면의 사람 검출 | `models/person.pt` |
| [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) | 얼굴 검출 → 자동 모자이크 (확장 기능) | `models/face.pt` |

## 설치와 실행 (Windows)

Python 3.11 이상을 권장합니다. 프로젝트 폴더의 PowerShell에서:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt
& ".\.venv\Scripts\python.exe" .\main.py
```

## GUI 사용법

1. **영상 열기**로 파일을 고르거나, 번호를 정하고 **카메라 연결**을 누릅니다.
2. **검출 모델**, **추적기**(ByteTrack 기본 / 개선 / 튜닝 설정), **대상**(사람·고양이·개·새), **얼굴 모자이크**를 고르고 **▶ 시작**을 누릅니다.
3. 화면에 박스·ID·이동 궤적이 그려지고, 가려지거나 화면 밖으로 나간 객체는 점선 회색 박스로 표시됩니다.
   오른쪽 목록에서 객체별 상태(추적 중·가림·화면 이탈·재등장·재진입), 이동 방향, 누적 거리, 속도를 확인합니다.
   위쪽에는 현재 FPS, 평균 FPS·평균 ms, 단계별(검출·추적·모자이크) 시간이 나옵니다.
4. **일시정지 → 시작**은 이어서, **중지 → 시작**은 처음부터 추적합니다.
5. 추적이 끝나면 **결과 저장**으로 CSV(+설정 JSON)를 저장하고, **결과 불러오기**로 모델 없이 다시 재생할 수 있습니다.
6. **설정 저장/불러오기**는 모델·추적기·대상·표시 옵션을 JSON으로 저장합니다.
7. **성능 비교** 탭은 `run_mot.py`, `benchmark_speed.py`가 만든 비교표를 보여 줍니다.

검출 모델 기본값은 `models/person.pt` → `models/coco.pt` → `yolo11n.pt` 순서로 고릅니다.
고양이·개·새는 COCO 클래스가 있는 가중치(`yolo11n.pt`, `models/coco.pt`)에서만 선택할 수 있습니다.

## 파일 구성

| 파일 | 역할 | 요구 사항 |
|---|---|---|
| `tracker/` | ByteTrack 직접 구현(칼만 필터, 헝가리안 연결, 상태 관리)과 개선 기능, 이동 정보 계산 | 2·3·4·5 |
| `pipeline.py` | 검출 → 외형 특징 → 추적 → 얼굴 모자이크를 묶은 처리 파이프라인 | 8 |
| `main.py`, `tracking.py` | GUI와 영상·카메라 처리 스레드 | 8·9 |
| `results_io.py` | 추적 결과 CSV/MOT 저장·불러오기 | 10 |
| `prepare_dataset.py` | 4개 데이터셋 train/val/test 분할과 전처리 | 1 |
| `train_models.py` | YOLO11n Fine-tuning, test 평가, 가중치·설정 설치 | 6·10 |
| `run_mot.py` | MOT 시퀀스에서 추적기별 성능 측정 → 비교표 | 3·7·9 |
| `tune_params.py` | 검출·연관성 임계값, 추적 유지 시간 격자 탐색 | 6 |
| `analyze_failures.py` | ID 변경·끊김 사례를 교차·가림·화면 경계로 분류, 사례 이미지 | 4·7 |
| `benchmark_speed.py` | 처리 시간·FPS와 속도 영향 요인 측정 | 8 |
| `metrics.py` | HOTA·MOTA·IDF1·IDSW 계산 (TrackEval과 같은 값 확인) | 7 |
| `evaluate_mot.py` | 공식 TrackEval 교차 확인 (선택) | 7 |
| `tests/` | 단위 테스트와 정답을 아는 합성 장면 | — |

## 전체 실행 순서

아래 명령은 PowerShell 기준이며 `python`은 `.\.venv\Scripts\python.exe`로 바꿔 실행하세요.

### 1) 데이터 준비 (요구 사항 1)

내려받은 원본은 모두 `datasets/raw/` 아래에 풉니다. `datasets/`는 `.gitignore`에 들어 있어 수십 GB 데이터가 git에 잡히지 않고,
전처리 결과(`datasets/mot`, `datasets/coco4`, `datasets/crowd`, `datasets/face`)와도 폴더가 겹치지 않습니다.

```
datasets/raw/
├─ MOT17/train/MOT17-02-FRCNN/{img1, gt, seqinfo.ini} ...   (MOT17.zip)
├─ MOT20/train/...                                          (MOT20.zip, 선택)
├─ coco/{train2017, val2017, annotations}                   (선택)
├─ crowd/{train/Images, val/Images, annotation_train.odgt, annotation_val.odgt}  (선택)
└─ wider/{WIDER_train, WIDER_val, wider_face_split}          (선택)
```

MOT17·MOT20은 공식 ZIP의 `train` 폴더를 사용합니다 (test는 정답이 공개되지 않음).

```powershell
python prepare_dataset.py mot --sequences datasets/raw/MOT17/train datasets/raw/MOT20/train --output datasets/mot
python prepare_dataset.py coco --train-images datasets/raw/coco/train2017 --train-labels datasets/raw/coco/annotations/instances_train2017.json --val-images datasets/raw/coco/val2017 --val-labels datasets/raw/coco/annotations/instances_val2017.json --output datasets/coco4
python prepare_dataset.py crowd --train-images datasets/raw/crowd/train/Images --train-labels datasets/raw/crowd/annotation_train.odgt --val-images datasets/raw/crowd/val/Images --val-labels datasets/raw/crowd/annotation_val.odgt --output datasets/crowd
python prepare_dataset.py face --train-images datasets/raw/wider/WIDER_train/images --train-labels datasets/raw/wider/wider_face_split/wider_face_train_bbx_gt.txt --val-images datasets/raw/wider/WIDER_val/images --val-labels datasets/raw/wider/wider_face_split/wider_face_val_bbx_gt.txt --output datasets/face
```

MOT20을 받지 않았으면 첫 줄에서 `datasets/raw/MOT20/train`을 빼세요.
CrowdHuman은 `CrowdHuman_train01~03.zip`의 이미지를 `datasets/raw/crowd/train/Images` 하나에 모으고, `CrowdHuman_val.zip`은 `datasets/raw/crowd/val/Images`로 풉니다.
COCO는 `--limit 5000`처럼 분할당 이미지 수를 제한해 빠르게 실험할 수 있습니다.
기본 `yolo11n.pt`가 이미 COCO로 학습된 모델이므로, 사람·고양이·개·새 검출은 COCO 학습 없이도 동작합니다.

### 2) 검출기 Fine-tuning (요구 사항 6)

```powershell
python train_models.py coco  --data datasets/coco4/dataset.yaml --weights yolo11n.pt --epochs 30 --install
python train_models.py crowd --data datasets/crowd/dataset.yaml --weights yolo11n.pt --epochs 30
python train_models.py mot   --data datasets/mot/dataset.yaml   --weights yolo11n.pt --epochs 30
python train_models.py face  --data datasets/face/dataset.yaml  --weights yolo11n.pt --epochs 30 --install
```

CUDA가 있으면 `--device 0`을 붙이고, `--workers 4`로 데이터 로딩을 병렬화하면 GPU를 더 잘 씁니다.
GPU가 없으면 아래 "Google Colab에서 학습하기"를 참고하세요. `--install` 없이 학습한 가중치는 아래 3)에서 `--weights`로 비교한 뒤
좋은 쪽을 `models/person.pt`로 복사하세요 (`--install`은 기존 파일을 날짜를 붙여 백업합니다).

### Google Colab에서 학습하기

[`colab/train_mot17.ipynb`](colab/train_mot17.ipynb)를 Colab에서 열고(파일 → 노트북 업로드, 또는 GitHub 탭에서 이 저장소 선택)
**런타임 → 런타임 유형 변경 → GPU**를 고른 뒤 위에서부터 차례로 실행합니다.
MOT17 내려받기, 전처리, Fine-tuning 전 성능 측정, 학습, 학습 전후 비교, 파라미터 튜닝, 최종 비교, 실패 분석까지 진행하고
결과(`models/person.pt`+`.json`, `configs/tuned.json`, `results/`)를 Google Drive의 `AI_Project_Level_3_outputs/`에 저장합니다.
이 폴더의 내용을 내 PC 프로젝트 폴더에 같은 경로로 복사하면 GUI의 모델 목록·추적기 목록·성능 비교 탭에 그대로 나타납니다.
처리 속도(요구 사항 8)는 발표 PC에서 `benchmark_speed.py`로 따로 측정하세요.

### 3) 추적 성능 비교와 튜닝 (요구 사항 3·6·7)

```powershell
# val 구간에서 파라미터 탐색 → configs/tuned.json
python tune_params.py datasets/mot/tracking/val --base improved --name tuned

# test 구간에서 기본판 / 개선판 / 튜닝판 비교 → results/comparison.json (GUI 성능 비교 탭)
python run_mot.py datasets/mot/tracking/test --trackers baseline improved tuned --name test

# 개선 항목별 효과 (ablation)
python run_mot.py datasets/mot/tracking/test --trackers baseline baseline+appearance baseline+recover_lost baseline+reid_buffer --name ablation

# 검출 모델 비교 (같은 추적기, 가중치만 다르게)
python run_mot.py datasets/mot/tracking/test --weights training_runs/crowd/weights/best.pt --trackers improved --name crowdhuman

# 가림·교차 실패 사례 분석 → results/failures/test/
python analyze_failures.py datasets/mot/tracking/test --trackers baseline improved --name test
```

### 4) 처리 속도 (요구 사항 8)

```powershell
python benchmark_speed.py sample.mp4 --frames 150
```

### 테스트

```powershell
python -m unittest discover -s tests -t .
```

## 검증한 내용과 한계

* `metrics.py`의 HOTA·DetA·AssA·MOTA·IDF1·IDSW·Frag가 공식 TrackEval과 소수점 셋째 자리까지 같음을 합성 시퀀스로 확인했습니다.
* 정답을 아는 합성 장면(교차·기둥 가림·화면 이탈 후 재진입)에서 개선판은 IDSW 1 → 0, IDF1 90.30 → 96.17, HOTA 84.00 → 87.23입니다.
  다만 이 장면에서 효과를 낸 것은 재진입 ID 복원뿐이었습니다. 외형 연결과 가림 후 재연결의 효과는 **MOT 데이터에서 확인해야 합니다.**
* 공식 MOT17/MOT20, COCO, CrowdHuman, WIDER FACE 전체 학습·평가 수치는 아직 없습니다. 위 3)을 실행해 `results/comparison.md`의 표를 보고서에 사용하세요.
* ID는 영상 안에서만 유효한 번호입니다. 외형 특징은 색 히스토그램이라 옷 색이 비슷한 사람끼리는 구분력이 떨어집니다.
* 얼굴 모자이크는 검출된 얼굴에만 적용되고, 원본 재생·결과 재생 모드에는 적용되지 않습니다.

### 이전 실험 기록

| 가중치 | 데이터 | 설정 | 검증 mAP50–95 |
|---|---|---|---:|
| `experiments/crowdhuman_person.pt` | CrowdHuman 96 / 32장 | 4에폭, 크기 256 | 0.154 |
| `models/face.pt` | WIDER FACE 96 / 32장 | 4에폭, 크기 384 | 0.197 |
| `experiments/coco128_1epoch.pt` | COCO128 96 / 32장 | 1에폭, 크기 256 | 0.359 (기본 YOLO11n 0.385) |

MOT17-02-FRCNN 첫 60프레임에서 Ultralytics 내장 ByteTrack으로 측정한 HOTA는 기본 YOLO11n 38.40%, CrowdHuman 소량 학습본 10.29%였습니다.
CrowdHuman 가중치(`models/person_crowdhuman.pt`)는 이 결과 때문에 기본값으로 쓰지 않으며, GUI 모델 목록에서 직접 고를 수 있습니다.
얼굴 가중치는 [akanametov/yolo-face](https://github.com/akanametov/yolo-face/releases/tag/1.0.0)의 YOLO11n 얼굴 가중치에서 시작했습니다.
