"""확장 기능: 보호색 동물 검출·추적 (담당: kumohyunho / ChatGPT).

계획 (예시, 담당자가 자유롭게 바꿔도 됩니다)
1. 데이터: MoCA(움직이는 보호색 동물 영상), COD10K·CAMO(마스크 → 박스 변환)
2. 검출: YOLO11n Fine-tuning → ``models/camouflage.pt``
3. 추적: 보호색 동물은 검출 신뢰도가 낮게 나오기 쉬우므로 ByteTrack의 2차(낮은 신뢰도) 연결이 중요합니다.
   ``tracker``의 ``high_thresh``·``low_thresh``·``new_track_thresh``를 낮춰 실험해 보세요.
4. 일반 동물(COCO 사전학습 yolo11n)과 성능 비교, 실패 사례 분석

``models/camouflage.pt``가 생기면 아래 설정만으로 GUI에서 바로 켤 수 있습니다.
"""
from .base import DetectTrackExtension


class CamouflageExtension(DetectTrackExtension):
    key = "camouflage"
    label = "보호색 동물"
    tag = "보"
    description = "배경과 색이 비슷한 동물을 검출·추적합니다."
    weights = "camouflage.pt"
    classes = None  # 가중치의 모든 클래스
    tracker = {"preset": "improved", "high_thresh": 0.4, "low_thresh": 0.05, "new_track_thresh": 0.5}
