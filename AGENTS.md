# AGENTS.md — AI 작업 지침 (Claude·ChatGPT/Codex 공통)

이 저장소는 두 사람이 서로 다른 AI로 함께 작업합니다. 코드를 바꾸기 전에 이 파일과 `docs/COLLABORATION.md`를 읽고 따르세요.

## 프로젝트
과제 「3.5 지능형 객체 추적 시스템」: YOLO11n 검출 + 직접 구현한 ByteTrack(기본판·개선판) + PySide6 GUI.
* 요구 사항 10개와 구현 대응: `docs/REQUIREMENTS.md`
* ByteTrack 원리·개선: `docs/BYTETRACK.md`
* 실행 방법·전체 실행 순서: `README.md`

## 역할 분담
| 담당 | AI | 작업 | 브랜치 | 주로 고치는 파일 |
|---|---|---|---|---|
| Pdint | Claude | 확장: 한글 글자 검출·추적·인식 | `claude/hangul` | `extensions/hangul.py`, 한글 데이터 전처리·학습 코드 |
| kumohyunho | ChatGPT (Codex) | 확장: 보호색 동물 검출·추적 | `feature/camouflage` (Codex는 `codex/...`도 가능) | `extensions/camouflage.py`, MoCA·COD10K 전처리·학습 코드 |

다른 사람이 맡은 확장 기능 파일은 고치지 마세요. 꼭 필요하면 PR 본문에 이유를 적고 담당자에게 확인받습니다.

## 반드시 지킬 규칙
1. **`main`에 직접 push하지 않습니다.** 작업 브랜치 → PR → 상대방 승인 → merge. (`main`은 보호 규칙이 걸려 있습니다.)
2. 작업을 시작하기 전에 최신 `main`을 받고(`git pull origin main`), 그 위에서 브랜치를 만듭니다.
3. 확장 기능은 `extensions/` 안에 **파일 하나 + `extensions/__init__.py`의 `REGISTRY` 한 줄**로 추가합니다.
   공통 파일(`pipeline.py`, `main.py`, `tracking.py`, `results_io.py`, `tracker/`)은 고치지 않는 것이 원칙입니다.
   공통 틀에 기능이 더 필요하면 **그 변경만 따로 작은 PR**로 올립니다 (확장 기능 PR과 섞지 않기).
4. 데이터셋은 `datasets/raw/`(원본)·`datasets/<이름>/`(전처리 결과)에 두고 **커밋하지 않습니다** (`.gitignore` 대상).
5. 가중치는 `models/<이름>.pt`와 학습 설정·성능 `models/<이름>.json`을 함께 커밋합니다 (`train_models.py --install` 형식). 파일 하나는 100MB 미만이어야 합니다.
6. 새 패키지가 필요하면 `requirements.txt`에 버전 범위와 함께 추가하고 PR 본문에 적습니다.
7. PR 전에 테스트를 실행합니다: `python -m unittest discover -s tests -t .`
   새 기능에는 모델 없이 돌아가는 테스트를 추가합니다 (예: `tests/test_extensions.py`의 가짜 검출기 방식).
8. 화면 문구·문서·커밋 메시지는 한국어로 씁니다.
9. 성능 수치는 실제로 실행해 얻은 값만 씁니다. 실행하지 못했으면 "미실행"이라고 적습니다.

## PR 본문에 쓸 것
* 무엇을 왜 바꿨는지 (요구 사항 번호 또는 확장 기능 이름)
* 실행한 테스트와 결과, 실행하지 못한 것
* 공통 파일을 고쳤다면 그 이유
