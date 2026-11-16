# 협업 방법

두 사람이 각자 다른 AI(Claude, ChatGPT/Codex)로 작업하지만, **GitHub 저장소 하나를 같은 원본으로** 씁니다.
AI 작업 규칙의 요약은 저장소 루트의 `AGENTS.md`에 있습니다 (Codex는 자동으로, Claude는 `CLAUDE.md`를 거쳐 읽습니다).

## 1. 작업 흐름

```bash
git checkout main
git pull origin main                 # 항상 최신 main에서 시작
git checkout -b claude/hangul        # 또는 feature/camouflage
# ... 작업 ...
python -m unittest discover -s tests -t .
git push -u origin claude/hangul
```

1. GitHub에서 PR을 엽니다 (base: `main`).
2. 상대방이 **Files changed → Review changes → Approve** 합니다. 상대방의 AI에게 PR 링크를 주고 리뷰를 맡겨도 됩니다.
3. 승인되면 **Merge pull request**. 내 브랜치가 뒤처졌다는 표시가 나오면 `git pull origin main`으로 합친 뒤 다시 push합니다.
4. 큰 기능은 한 번에 올리지 말고 "데이터 전처리 → 학습 → GUI 연결"처럼 나눠 PR을 올리면 리뷰와 충돌 해결이 쉽습니다.

## 2. 확장 기능 만드는 법

확장 기능 하나 = `extensions/` 아래 파일 하나입니다. 파이프라인·GUI·결과 파일은 등록 정보만 보고 자동으로 동작합니다.

* GUI 실시간 추적 탭의 **확장** 체크박스로 켜고 끕니다 (가중치가 없으면 비활성, 마우스를 올리면 이유 표시).
* 켜면 매 프레임 기본 추적(사람·동물) 뒤에 실행되고, 결과는 같은 화면·객체 목록·결과 CSV에 함께 나옵니다.
  ID 앞에 확장 기능 표시(`tag`)가 붙고(예: `글3`, `보2`), 인식한 글자 등은 ID 옆에 보입니다.
* 결과 CSV에는 `source`(확장 기능 key)와 `label`(붙인 글자) 열이 저장되고, 결과 재생에서도 그대로 보입니다.
* 설정 저장/불러오기에 켜 둔 확장 기능 목록이 함께 저장됩니다.
* 처리 시간은 상태 패널의 "확장 ○ ms"로 따로 표시됩니다.

### 전용 검출기 + 추적 (대부분의 경우)

```python
# extensions/camouflage.py
from .base import DetectTrackExtension

class CamouflageExtension(DetectTrackExtension):
    key = "camouflage"            # REGISTRY 키와 같게
    label = "보호색 동물"          # GUI 체크박스 이름
    tag = "보"                    # 화면 ID 앞 표시
    weights = "camouflage.pt"     # models/ 아래 가중치
    classes = None                # None = 가중치의 모든 클래스
    tracker = {"preset": "improved", "high_thresh": 0.4, "low_thresh": 0.05}

    def describe(self, frame, row):   # (선택) 트랙별 글자, 예: OCR 결과
        return ""
```

`extensions/__init__.py`의 `REGISTRY`에 `"camouflage": "extensions.camouflage:CamouflageExtension"`이 있으면 끝입니다.
`models/camouflage.pt`를 넣는 순간 GUI 체크박스가 활성화됩니다.

### 구조가 다른 경우
`extensions.base.Extension`을 직접 상속하고 `check()`(사용 불가 이유 또는 `None`)와
`process(frame, frame_index)`(화면에 그릴 `pipeline.TrackRow` 목록, `source=self.key`)만 구현하면 됩니다.
`frame`은 얼굴 모자이크 전 원본이며, 확장 기능이 직접 고치면 안 됩니다.

## 3. 파일 담당

| 파일 | 담당 | 비고 |
|---|---|---|
| `extensions/hangul.py`, 한글 데이터·학습 코드 | Pdint (Claude) | |
| `extensions/camouflage.py`, 보호색 데이터·학습 코드 | kumohyunho (ChatGPT) | |
| `extensions/base.py`, `extensions/__init__.py` | 공통 | 바꿀 때는 따로 작은 PR |
| `pipeline.py`, `main.py`, `tracking.py`, `results_io.py`, `tracker/` | 공통 | 바꿀 때는 따로 작은 PR |
| `prepare_dataset.py`, `train_models.py` | 공통 | 새 데이터 종류는 하위 명령 추가로 (기존 명령은 그대로) |
| `docs/REQUIREMENTS.md`, `README.md` | 공통 | 각자 자기 확장 기능 절만 고치기 |

## 4. 데이터·가중치

* 원본 데이터: `datasets/raw/<이름>/`, 전처리 결과: `datasets/<이름>/` — 둘 다 커밋하지 않습니다.
* 학습은 Colab을 써도 됩니다 (`colab/train_mot17.ipynb` 참고). 결과 가중치는 `models/<이름>.pt` + `models/<이름>.json`으로 커밋합니다.
* 같은 이름의 가중치를 덮어쓸 때는 PR 본문에 이전·새 성능을 함께 적습니다.

## 5. 다른 AI에게 일을 맡길 때 붙여 넣을 문장

> 저장소 `kumohyunho/AI_Project_Level_3-main`의 `AGENTS.md`와 `docs/COLLABORATION.md`를 먼저 읽고 규칙을 따라 줘.
> 나는 (한글 인식 / 보호색 동물) 확장 기능 담당이고, 작업은 `(브랜치 이름)` 브랜치에서 하고 PR로 올려 줘.
