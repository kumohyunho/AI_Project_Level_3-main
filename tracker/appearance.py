"""가벼운 외형 특징 추출 (요구 사항 3: 특징 추출 개선).

박스를 위·아래 절반(상의·하의)으로 나누고 각각 HSV 색 히스토그램을 구해 이어 붙입니다.
CPU에서도 프레임당 수 ms 이내로 계산되며, 별도 학습이 필요 없습니다.
가림 후 재등장, 화면 이탈 후 재진입 시 같은 사람인지 판단하는 데 사용합니다.
"""
import numpy as np

_BINS = (16, 4)  # Hue, Saturation
_RANGES = [0, 180, 0, 256]


def _histogram(region):
    import cv2

    if region.size == 0:
        return np.zeros(_BINS[0] * _BINS[1], dtype=np.float32)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # 너무 어둡거나 채도가 낮은 픽셀(그림자, 배경 아스팔트)은 색 정보가 약해서 제외합니다.
    mask = cv2.inRange(hsv, (0, 30, 40), (180, 255, 255))
    return cv2.calcHist([hsv], [0, 1], mask, list(_BINS), _RANGES).flatten()


def extract_features(frame, boxes):
    """``boxes``(x1y1x2y2) 각각의 L2 정규화된 128차원 특징. 너무 작은 박스는 ``None``."""
    if frame is None:
        return [None] * len(boxes)
    height, width = frame.shape[:2]
    features = []
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        # 테두리의 배경이 섞이지 않도록 박스 안쪽 80%만 사용합니다.
        dx, dy = (x2 - x1) * 0.1, (y2 - y1) * 0.1
        left, top = int(max(0, x1 + dx)), int(max(0, y1 + dy))
        right, bottom = int(min(width, x2 - dx)), int(min(height, y2 - dy))
        if right - left < 4 or bottom - top < 8:
            features.append(None)
            continue
        middle = (top + bottom) // 2
        vector = np.concatenate([
            _histogram(frame[top:middle, left:right]),
            _histogram(frame[middle:bottom, left:right]),
        ]).astype(np.float32)
        norm = np.linalg.norm(vector)
        features.append(vector / norm if norm > 0 else None)
    return features
