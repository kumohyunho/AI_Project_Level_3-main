"""검출과 트랙을 연결하는 비용 계산과 헝가리안 할당 (요구 사항 2·3: 객체 간 연관성 판단)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(boxes_a, boxes_b):
    """``(N,4)``와 ``(M,4)`` x1y1x2y2 박스의 IoU 행렬 ``(N,M)``."""
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def iou_distance(tracks, detections):
    return 1.0 - iou_matrix([t.box for t in tracks], [d.box for d in detections])


def fuse_score(cost, detections):
    """ByteTrack 방식: 신뢰도가 높은 검출일수록 같은 IoU에서도 비용을 낮춥니다."""
    if not cost.size:
        return cost
    scores = np.array([d.score for d in detections])[None, :]
    return 1.0 - (1.0 - cost) * scores


def appearance_distance(tracks, detections):
    """코사인 거리. 외형 특징이 없는 쪽은 중립값 0.5로 둡니다."""
    cost = np.full((len(tracks), len(detections)), 0.5)
    for row, track in enumerate(tracks):
        if track.feature is None:
            continue
        for column, detection in enumerate(detections):
            if detection.feature is not None:
                cost[row, column] = 1.0 - float(np.dot(track.feature, detection.feature))
    return cost


def class_gate(cost, tracks, detections):
    """다른 클래스(예: 사람과 개)끼리는 절대 연결하지 않습니다."""
    if cost.size:
        mismatch = np.array([t.cls for t in tracks])[:, None] != np.array(
            [d.cls for d in detections])[None, :]
        cost = np.where(mismatch, np.inf, cost)
    return cost


def linear_assignment(cost, threshold):
    """비용 합이 최소가 되도록 1:1로 짝짓고, 비용이 ``threshold``를 넘는 짝은 버립니다."""
    rows, columns = cost.shape if cost.ndim == 2 else (0, 0)
    if not rows or not columns:
        return [], list(range(rows)), list(range(columns))
    bounded = np.where(np.isfinite(cost), cost, 1e6)
    bounded = np.where(bounded > threshold, threshold + 1e3, bounded)
    matched_rows, matched_columns = linear_sum_assignment(bounded)
    matches = [(r, c) for r, c in zip(matched_rows, matched_columns) if cost[r, c] <= threshold]
    used_rows = {r for r, _ in matches}
    used_columns = {c for _, c in matches}
    return (matches, [r for r in range(rows) if r not in used_rows],
            [c for c in range(columns) if c not in used_columns])
