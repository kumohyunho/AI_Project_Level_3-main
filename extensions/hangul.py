"""확장 기능: 떠다니는 한글 자음·모음 검출·추적 (담당: Pdint / Claude).  설계: ``docs/HANGUL.md``

자음·모음 **하나하나가 추적 대상**입니다. 검출기가 40개 자모를 바로 분류하므로 OCR은 쓰지 않습니다.

1. 검출: 합성 데이터(``hangul_dataset.py``)로 YOLO11n Fine-tuning → ``models/hangul_jamo.pt`` (40클래스)
2. 추적: 자모는 회전하면 다른 글자로 보이므로(ㅏ↔ㅓ, ㅗ↔ㅜ) 클래스와 무관하게 위치·움직임으로 연결
3. 분류: 트랙별로 여러 프레임의 검출 클래스를 모아 다수결로 자모를 정해 표시

아직 2·3단계 구현 전입니다. ``check()``가 이유를 돌려주므로 GUI에는 비활성 체크박스로만 보입니다.
"""
from .base import DetectTrackExtension

# (클래스 이름, 글자, 종류). 순서가 곧 검출기의 클래스 번호입니다. 바꾸면 가중치를 다시 학습해야 합니다.
# 클래스 이름은 국어의 로마자 표기법에 따른 자모 이름입니다 (학습 도구가 ASCII 이름을 가장 안정적으로 처리).
JAMO = [
    ("giyeok", "ㄱ", "자음"), ("nieun", "ㄴ", "자음"), ("digeut", "ㄷ", "자음"), ("rieul", "ㄹ", "자음"),
    ("mieum", "ㅁ", "자음"), ("bieup", "ㅂ", "자음"), ("siot", "ㅅ", "자음"), ("ieung", "ㅇ", "자음"),
    ("jieut", "ㅈ", "자음"), ("chieut", "ㅊ", "자음"), ("kieuk", "ㅋ", "자음"), ("tieut", "ㅌ", "자음"),
    ("pieup", "ㅍ", "자음"), ("hieut", "ㅎ", "자음"),
    ("ssanggiyeok", "ㄲ", "자음"), ("ssangdigeut", "ㄸ", "자음"), ("ssangbieup", "ㅃ", "자음"),
    ("ssangsiot", "ㅆ", "자음"), ("ssangjieut", "ㅉ", "자음"),
    ("a", "ㅏ", "모음"), ("ya", "ㅑ", "모음"), ("eo", "ㅓ", "모음"), ("yeo", "ㅕ", "모음"), ("o", "ㅗ", "모음"),
    ("yo", "ㅛ", "모음"), ("u", "ㅜ", "모음"), ("yu", "ㅠ", "모음"), ("eu", "ㅡ", "모음"), ("i", "ㅣ", "모음"),
    ("ae", "ㅐ", "모음"), ("yae", "ㅒ", "모음"), ("e", "ㅔ", "모음"), ("ye", "ㅖ", "모음"), ("wa", "ㅘ", "모음"),
    ("wae", "ㅙ", "모음"), ("oe", "ㅚ", "모음"), ("wo", "ㅝ", "모음"), ("we", "ㅞ", "모음"), ("wi", "ㅟ", "모음"),
    ("ui", "ㅢ", "모음"),
]
CLASS_NAMES = [name for name, _, _ in JAMO]
CHAR = {name: char for name, char, _ in JAMO}
KIND = {name: kind for name, _, kind in JAMO}


class HangulJamoExtension(DetectTrackExtension):
    key = "hangul"
    label = "한글 자모"
    tag = "자"
    description = "떠다니는 한글 자음·모음을 하나씩 검출·추적하고 어떤 자모인지 표시합니다."
    weights = "hangul_jamo.pt"
    tracker = "baseline"

    @classmethod
    def check(cls):
        reason = super().check()
        return reason or "구현 중입니다 (claude/hangul 브랜치)."

    def describe(self, frame, row):
        return CHAR.get(row.cls_name, "")  # TODO: 트랙별 다수결
