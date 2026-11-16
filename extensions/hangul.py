"""확장 기능: 영상 속 한글 글자 검출·추적·인식 (담당: Pdint / Claude).

계획
1. 글자 영역 검출: 한글 글자 영역 데이터로 YOLO11n을 Fine-tuning → ``models/hangul_text.pt``
2. 추적: 움직이는 글자(간판·자막·떠다니는 글자)에 ByteTrack으로 ID 부여
3. 인식: 트랙별로 잘라낸 영역을 한글 OCR로 읽고, 여러 프레임의 결과를 모아(다수결) 안정적인 문장을 표시

아직 구현 전입니다. ``check()``가 이유를 돌려주므로 GUI에는 비활성 체크박스로만 보입니다.
"""
from .base import DetectTrackExtension


class HangulTextExtension(DetectTrackExtension):
    key = "hangul"
    label = "한글 인식"
    tag = "글"
    description = "영상 속 한글 글자 영역을 검출·추적하고 OCR로 읽습니다."
    weights = "hangul_text.pt"
    tracker = "baseline"

    @classmethod
    def check(cls):
        reason = super().check()
        return reason or "구현 중입니다 (claude/hangul 브랜치)."

    def describe(self, frame, row):
        return ""  # TODO: OCR 결과 (트랙별 다수결)
