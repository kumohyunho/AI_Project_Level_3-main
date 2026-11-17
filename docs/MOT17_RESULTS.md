# MOT17 보행자 추적 결과 (요구 사항 1·3·6·7)

기본 시스템(보행자 추적)을 공개 데이터 MOT17로 평가한 결과입니다. **검출기 Fine-tuning으로 HOTA가 40.88에서 51.46으로
10.6 올랐고**, 추적기 개선은 검증 구간에서는 효과가 있었지만 테스트 구간에서는 재현되지 않았습니다.

## 데이터와 분할

| 항목 | 내용 |
|---|---|
| 데이터 | MOT17 학습용 영상 7개 (FRCNN 버전: 02, 04, 05, 09, 10, 11, 13). 공식 test는 정답이 공개되지 않아 쓰지 않음 |
| 분할 | 영상마다 시간 순으로 앞 50% = train, 다음 25% = val, 마지막 25% = test (`prepare_dataset.py mot`) |
| 프레임 수 | train 2,657 / val 1,327 / test 1,332 |
| 검출 학습 이미지 | train 533장 (5프레임 간격, 30% 이상 보이는 보행자), val·test 각 268장 |
| 사용 원칙 | 학습은 train, 설정 선택은 val, 최종 보고는 test |

## 1. 검출기 Fine-tuning (요구 사항 6)

COCO 사전학습 `yolo11n.pt`를 MOT17 train 구간으로 30에폭 학습했습니다 (Colab T4, AdamW, 학습률 0.0005,
앞 10개 레이어 고정, 입력 640, 배치 32). 결과 모델은 `models/person.pt`(+ `person.json`)로 설치되어 GUI·평가의 기본 검출기가 됩니다.

| 검출기 (test 268장) | mAP50 | mAP50-95 | 정밀도 | 재현율 |
|---|---:|---:|---:|---:|
| 기본 `yolo11n.pt` (사람 클래스) | 0.693 | 0.437 | 0.804 | 0.580 |
| MOT17 학습 모델 | **0.860** | **0.547** | **0.864** | **0.750** |

* 재현율이 0.58 → 0.75로 가장 크게 올랐습니다. 기본 모델이 놓치던 작거나 멀리 있는 보행자를 잡게 되었습니다.
* Colab에서 기록한 test 성능(mAP50 0.8597)과 이 저장소의 test 구간에서 다시 잰 값(0.860)이 같아, 같은 분할로 학습·평가했음을 확인했습니다.

## 2. 추적 성능 (요구 사항 3·7)

같은 test 구간 1,332프레임에서 `run_mot.py`로 측정했습니다 (`results/comparison.md`, GUI 성능 비교 탭).

| 검출기 | 추적기 | HOTA | IDF1 | MOTA | ID 변경 | 끊김 |
|---|---|---:|---:|---:|---:|---:|
| 기본 `yolo11n.pt` | 기본판 | 40.88 | 43.95 | 29.02 | 44 | 78 |
| 기본 `yolo11n.pt` | 개선판 | 41.10 | 43.92 | 29.21 | 47 | 82 |
| 기본 `yolo11n.pt` | 튜닝판 (`configs/tuned.json`) | 42.88 | 47.73 | 31.81 | 65 | 105 |
| MOT17 학습 모델 | 기본판 | **51.46** | **61.89** | **48.35** | 82 | 133 |
| MOT17 학습 모델 | 개선판 | 51.03 | 61.16 | 47.49 | 115 | 185 |
| MOT17 학습 모델 | 튜닝판 (`configs/tuned_person.json`) | 50.91 | 61.28 | 47.97 | 113 | 168 |

* **가장 큰 개선은 검출기 Fine-tuning입니다.** 기본판끼리 비교하면 HOTA +10.6, IDF1 +17.9, MOTA +19.3.
* ID 변경 횟수는 44 → 82회로 늘었습니다. 검출되는 사람이 많아져(재현율 0.58 → 0.75) 추적 대상과 교차 상황 자체가 늘었기 때문으로 보며,
  비율로 보는 IDF1은 크게 올랐습니다.
* 기본 검출기에서는 튜닝판이 가장 좋았습니다 (HOTA +2.0). 검출 임계값을 0.5 → 0.4로 낮춰 더 많은 사람을 잡은 효과입니다.

### 추적기 설정 선택 (val 구간)

| 검출기 | 추적기 | val HOTA | val IDF1 | val ID 변경 |
|---|---|---:|---:|---:|
| 기본 `yolo11n.pt` | 기본판 | 37.62 | 40.23 | 37 |
| 기본 `yolo11n.pt` | 개선판 | 38.27 | 40.79 | 43 |
| 기본 `yolo11n.pt` | 튜닝판 | **40.28** | **45.30** | 57 |
| MOT17 학습 모델 | 기본판 | 47.29 | 56.66 | 88 |
| MOT17 학습 모델 | + 가림 후 재연결 | 48.28 | 58.64 | 81 |
| MOT17 학습 모델 | + 재연결 + 재진입 ID 복원 | 48.31 | 58.83 | 75 |
| MOT17 학습 모델 | 개선판 | 48.25 | 58.46 | 117 |
| MOT17 학습 모델 | 튜닝판 | **49.92** | **60.77** | 94 |

* 튜닝판은 val 구간에서 27개 조합(검출 임계값 0.4/0.5/0.6 × 연관성 임계값 0.7/0.8/0.9 × 유지 시간 30/60/90프레임) 중
  HOTA가 가장 높은 조합입니다. 기본 검출기는 (0.4, 0.8, 60), 학습 모델은 (0.5, 0.7, 90)이 골라졌습니다.
* **학습 모델에서는 val에서 고른 튜닝판이 test에서 기본판보다 낮았습니다** (val +2.6 → test −0.6).
  두 구간의 차이가 1~3점 안쪽이라 설정 선택의 효과보다 구간별 장면 차이가 더 큰 것으로 보입니다. 설정은 val로만 골랐고 test로 다시 고르지 않았습니다.

### 개선 항목별 효과 (test, 기본판에 하나씩 추가)

| 추가한 항목 | 기본 검출기 HOTA | ID 변경 | 학습 모델 HOTA | ID 변경 |
|---|---:|---:|---:|---:|
| 없음 (기본판) | 40.88 | 44 | 51.46 | 82 |
| 외형 연결 | 41.16 | 49 | 51.22 | 114 |
| 가림 후 재연결 | 40.95 | **36** | **51.76** | 80 |
| 재진입 ID 복원 | 40.54 | 43 | 51.45 | **77** |
| 화면 밖 위치 고정 | 40.61 | 51 | 51.20 | 93 |
| 전부 (개선판) | 41.10 | 47 | 51.03 | 115 |

* **가림 후 재연결**은 두 검출기 모두에서 도움이 되었습니다 (기본 검출기 ID 변경 44 → 36회, 학습 모델 HOTA +0.3).
* **외형(색 히스토그램) 연결은 ID 변경을 늘렸습니다** (학습 모델 82 → 114회). 색 히스토그램만으로는
  비슷한 옷 색의 사람을 구분하기 어려운 것으로 보입니다 (원인을 장면별로 확인하지는 않았습니다). 한글 자모에서도 같은 결과가 나왔습니다 (`docs/HANGUL.md`).
* 개선판은 외형 연결의 손해 때문에 기본판보다 낮았습니다. 이 데이터에서는 "가림 후 재연결 + 재진입 ID 복원"만 쓰는 편이 낫습니다.

## 3. 실패 사례 (요구 사항 4·7)

`analyze_failures.py`로 test 구간의 ID 변경을 원인별로 나눴습니다 (`results/failures/test`, `results/failures/test_finetuned`).
분류 기준: 교차 = 다른 사람과 IoU 0.3 이상, 가림 = 정답의 가시 비율 0.5 미만, 화면 경계 = 가장자리 3% 이내.

| 검출기 | 추적기 | 교차 | 가림 | 화면 경계 | 기타 | 합계 |
|---|---|---:|---:|---:|---:|---:|
| 기본 `yolo11n.pt` | 기본판 | 21 | 9 | 2 | 12 | 44 |
| 기본 `yolo11n.pt` | 개선판 | 19 | 4 | 7 | 17 | 47 |
| MOT17 학습 모델 | 기본판 | 36 | 14 | 3 | 29 | 82 |
| MOT17 학습 모델 | 개선판 | 58 | 18 | 10 | 29 | 115 |
| MOT17 학습 모델 | 튜닝판 | 53 | 21 | 13 | 26 | 113 |

* **보행자에서는 교차가 ID 변경의 가장 큰 원인**입니다 (기본판 44~48%). 사람끼리 스쳐 지나갈 때 박스가 겹쳐 IoU만으로 구분하기 어렵습니다.
* 개선판이 교차 실패를 오히려 늘린 것(36 → 58회)은 외형 연결 때문으로 보입니다. 비슷한 옷 색의 두 사람이 스칠 때 외형 거리가 서로 바뀐 짝을 지지합니다.
* 개선판은 기본 검출기에서 가림으로 인한 ID 변경을 9 → 4회로 줄였습니다 (가림 후 재연결의 효과).
* 사례 그림은 추적기별 폴더에 12장씩 있습니다 (정답 흰색, 이전 ID 빨강, 새 ID 초록).

## 한계

* train·val·test가 **같은 7개 영상**을 시간 순으로 나눈 것이라, 테스트 장면의 배경·카메라가 학습 때와 같습니다.
  Fine-tuning의 효과가 처음 보는 장소의 영상보다 크게 나왔을 수 있습니다.
* MOT17 FRCNN 7개 영상만 썼고, MOT20(밀집 장면)과 공식 test 세트로는 평가하지 않았습니다.
* 처리 속도(FPS)는 개발 환경 CPU 값입니다 (검출 약 52 ms/프레임). 발표 PC에서 다시 재야 합니다.
* 외형 특징이 색 히스토그램이라 같은 색 옷의 사람을 구분하지 못합니다. 학습된 재식별(ReID) 특징은 쓰지 않았습니다.

## 재현

```powershell
python prepare_dataset.py mot --sequences datasets/raw/MOT17/train --output datasets/mot
python train_models.py mot --data datasets/mot/dataset.yaml --weights yolo11n.pt --epochs 30 --install   # GPU 권장 (colab/train_mot17.ipynb)
python tune_params.py datasets/mot/tracking/val --base improved --name tuned_person --weights models/person.pt
python run_mot.py datasets/mot/tracking/test --weights models/person.pt --trackers baseline improved tuned_person --name finetuned
python run_mot.py datasets/mot/tracking/test --weights models/person.pt --name ablation_finetuned --trackers baseline baseline+appearance baseline+recover_lost baseline+reid_buffer baseline+freeze_out_of_frame improved
python analyze_failures.py datasets/mot/tracking/test --weights models/person.pt --trackers baseline improved tuned_person --name test_finetuned --images 12
```

평가는 CPU에서 했고, 학습만 Colab T4 GPU에서 했습니다. 학습 원본 가중치와 설정은 `experiments/mot17/`에도 있습니다.
