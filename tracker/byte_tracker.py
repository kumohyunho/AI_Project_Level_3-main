"""ByteTrack 직접 구현과 개선판 (요구 사항 2·3·4).

기본판(``baseline``)은 원 논문(Zhang et al., "ByteTrack", ECCV 2022)의 절차를 그대로 따릅니다.

1. 검출을 신뢰도로 **높은 것 / 낮은 것**으로 나눕니다.
2. 기존 트랙의 위치를 칼만 필터로 예측합니다.
3. **1차 연결**: 높은 신뢰도 검출 ↔ (추적 중 + 잃어버린) 트랙을 IoU 비용으로 헝가리안 할당합니다.
4. **2차 연결**: 1차에서 남은 추적 중 트랙 ↔ 낮은 신뢰도 검출. 가려져 점수가 떨어진 사람을 놓치지 않는 ByteTrack의 핵심입니다.
5. 남은 트랙은 Lost 상태가 되고 ``track_buffer`` 프레임 동안 다시 나타나기를 기다립니다.
6. 연결되지 않은 높은 신뢰도 검출은 새 트랙이 되고, 다음 프레임에도 연결되어야 확정됩니다.

개선판(``improved``)은 다음을 추가합니다. 각각 설정으로 켜고 끌 수 있어 효과를 따로 측정할 수 있습니다.

* ``appearance``: 1차 연결 비용에 색 히스토그램 외형 거리를 섞어, 교차하는 두 사람의 ID가 바뀌는 것을 줄입니다.
* ``recover_lost``: IoU가 0이 될 만큼 오래 가려졌던 트랙을 외형 + 이동 거리 조건으로 다시 찾습니다.
* ``reid_buffer``: 화면 밖으로 나가 삭제된 트랙의 외형을 일정 시간 보관하고, 비슷한 사람이 다시 들어오면 이전 ID를 돌려줍니다.
* ``freeze_out_of_frame``: 화면 밖으로 나간 트랙의 예측 위치를 고정해 엉뚱한 곳으로 흘러가지 않게 합니다.
* ``damp_lost_velocity`` (실험용, 기본 꺼짐): 가려진 동안 예측 속도를 줄입니다. 등속으로 걷는 사람에게는 오히려 해로워 기본 설정에서 뺐습니다.
"""
from dataclasses import asdict, dataclass, fields

import numpy as np

from .kalman import KalmanFilter
from .matching import (appearance_distance, class_gate, fuse_score, iou_distance,
                       iou_matrix, linear_assignment)


class TrackState:
    NEW, TRACKED, LOST, REMOVED = range(4)


# 화면과 결과 파일에 쓰는 상태 이름
STATUS_TRACKED = "추적 중"
STATUS_REAPPEARED = "재등장"
STATUS_REENTERED = "재진입"
STATUS_OCCLUDED = "가림"
STATUS_EXITED = "화면 이탈"


@dataclass
class TrackerConfig:
    high_thresh: float = 0.5          # 1차 연결에 쓰는 검출 신뢰도
    low_thresh: float = 0.1           # 2차 연결에 쓰는 검출 신뢰도 하한
    new_track_thresh: float = 0.6     # 새 트랙을 만들 최소 신뢰도
    match_thresh: float = 0.8         # 1차 연결 비용 상한 (1 - IoU 기반)
    second_match_thresh: float = 0.5
    unconfirmed_match_thresh: float = 0.7
    track_buffer: int = 30            # Lost 트랙 유지 시간 (30fps 기준 프레임)
    min_box_area: float = 10.0
    appearance: bool = False
    appearance_weight: float = 0.3
    appearance_gate: float = 0.4
    feature_momentum: float = 0.9
    recover_lost: bool = False
    recover_thresh: float = 0.25
    reid_buffer: int = 0              # 삭제 트랙 외형 보관 시간 (0이면 끔)
    reid_thresh: float = 0.2
    freeze_out_of_frame: bool = False
    damp_lost_velocity: bool = False

    @classmethod
    def preset(cls, name):
        if name == "baseline":
            return cls()
        if name == "improved":
            # damp_lost_velocity는 합성 장면 ablation에서 단독 사용 시 HOTA를 떨어뜨려(84.0 → 71.2)
            # 기본 개선 설정에서 제외했습니다. 실험용 옵션으로만 남겨 둡니다.
            return cls(appearance=True, recover_lost=True, reid_buffer=150, freeze_out_of_frame=True)
        raise ValueError(f"알 수 없는 추적기 설정: {name}")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        names = {field.name: field.type for field in fields(cls)}
        unknown = set(data) - set(names)
        if unknown:
            raise ValueError(f"알 수 없는 추적기 설정 항목: {', '.join(sorted(unknown))}")
        return cls(**data)


def _to_xyah(box):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return np.array([x1 + width / 2, y1 + height / 2, width / max(height, 1e-6), height])


class STrack:
    def __init__(self, box, score, cls, feature=None):
        self._box = np.asarray(box, dtype=float)
        self.score = float(score)
        self.cls = int(cls)
        self.feature = feature
        self.mean = self.covariance = None
        self.track_id = 0
        self.state = TrackState.NEW
        self.is_activated = False
        self.start_frame = self.frame_id = self.lost_frame = 0
        self.tracklet_len = 0
        self.status = STATUS_TRACKED
        self.status_since = 0
        self.out_of_frame = False
        self.clean_feature = True   # 다른 검출과 겹치지 않은 깨끗한 외형인지

    @property
    def box(self):
        if self.mean is None:
            return self._box.copy()
        cx, cy, aspect, height = self.mean[:4]
        width = aspect * height
        return np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])

    def predict(self, kalman, damp=False, freeze=False):
        if freeze and self.out_of_frame:
            self.mean[4:] = 0
            return
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[7] = 0  # 가려진 동안 높이 변화는 예측하지 않습니다.
            if damp:
                mean[4:7] *= 0.85
        self.mean, self.covariance = kalman.predict(mean, self.covariance)

    def _update_feature(self, feature, momentum):
        if feature is None:
            return
        if self.feature is None:
            self.feature = feature
        else:
            mixed = momentum * self.feature + (1 - momentum) * feature
            self.feature = mixed / max(np.linalg.norm(mixed), 1e-9)

    def activate(self, kalman, frame_id, track_id):
        self.track_id = track_id
        self.mean, self.covariance = kalman.initiate(_to_xyah(self._box))
        self.state = TrackState.TRACKED
        self.is_activated = frame_id == 1
        self.start_frame = self.frame_id = frame_id
        self.tracklet_len = 0
        self.status, self.status_since = STATUS_TRACKED, frame_id

    def update(self, kalman, detection, frame_id, momentum, status=None):
        self.mean, self.covariance = kalman.update(
            self.mean, self.covariance, _to_xyah(detection._box))
        if detection.clean_feature:
            self._update_feature(detection.feature, momentum)
        was_tracked = self.state == TrackState.TRACKED
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = detection.score
        self.frame_id = frame_id
        self.tracklet_len = self.tracklet_len + 1 if was_tracked else 0
        self.out_of_frame = False
        if status:
            self.status, self.status_since = status, frame_id
        elif self.status in (STATUS_REAPPEARED, STATUS_REENTERED) and frame_id - self.status_since > 30:
            self.status = STATUS_TRACKED  # '재등장' 표시는 약 1초간 유지합니다.

    def mark_lost(self, frame_id, frame_size):
        self.state = TrackState.LOST
        self.lost_frame = frame_id
        self.out_of_frame = frame_size is not None and _near_border(self.box, self.mean, frame_size)
        self.status = STATUS_EXITED if self.out_of_frame else STATUS_OCCLUDED
        self.status_since = frame_id


def _near_border(box, mean, frame_size):
    """마지막 박스가 화면 가장자리에 닿아 있고 바깥쪽으로 움직이고 있었으면 화면 이탈로 봅니다."""
    width, height = frame_size
    x1, y1, x2, y2 = box
    margin_x, margin_y = 0.02 * width, 0.02 * height
    vx, vy = mean[4], mean[5]
    left = x1 <= margin_x and vx <= 0
    right = x2 >= width - margin_x and vx >= 0
    top = y1 <= margin_y and vy <= 0
    bottom = y2 >= height - margin_y and vy >= 0
    center_x, center_y = mean[0], mean[1]
    outside = not (0 <= center_x <= width and 0 <= center_y <= height)
    return outside or left or right or top or bottom


class ByteTracker:
    def __init__(self, config=None, frame_rate=30.0):
        self.config = config or TrackerConfig()
        self.kalman = KalmanFilter()
        scale = (frame_rate or 30.0) / 30.0
        self.max_time_lost = max(1, int(round(scale * self.config.track_buffer)))
        self.reid_time = int(round(scale * self.config.reid_buffer))
        self.reset()

    def reset(self):
        self.frame_id = 0
        self.next_id = 1
        self.tracked, self.lost, self.gallery = [], [], []

    # ------------------------------------------------------------------ helpers
    def _fused_cost(self, tracks, detections):
        """1차 연결 비용: IoU 거리(+ 외형 거리)에 검출 신뢰도를 반영합니다."""
        cost = iou_distance(tracks, detections)
        no_overlap = cost >= 1.0
        if self.config.appearance and cost.size:
            appearance = appearance_distance(tracks, detections)
            valid = np.array([[t.feature is not None and d.feature is not None for d in detections]
                              for t in tracks])
            weight = self.config.appearance_weight
            cost = np.where(valid, (1 - weight) * cost + weight * appearance, cost)
            cost = np.where(valid & (appearance > self.config.appearance_gate), np.inf, cost)
        cost = fuse_score(cost, detections)
        cost = np.where(no_overlap, np.inf, cost)
        return class_gate(cost, tracks, detections)

    def _recover_cost(self, tracks, detections):
        """오래 가려진 트랙 재연결 비용: 외형이 비슷하고 예측 위치 근처이며 크기가 비슷해야 합니다."""
        cost = appearance_distance(tracks, detections)
        for row, track in enumerate(tracks):
            if track.feature is None:
                cost[row, :] = np.inf
                continue
            gap = self.frame_id - track.lost_frame
            tx1, ty1, tx2, ty2 = track.box
            t_height = max(ty2 - ty1, 1.0)
            radius = min(3.0, 0.5 + 0.05 * gap) * t_height
            for column, detection in enumerate(detections):
                dx1, dy1, dx2, dy2 = detection._box
                d_height = max(dy2 - dy1, 1.0)
                distance = np.hypot((tx1 + tx2 - dx1 - dx2) / 2, (ty1 + ty2 - dy1 - dy2) / 2)
                ratio = d_height / t_height
                if detection.feature is None or distance > radius or not 0.6 <= ratio <= 1.6:
                    cost[row, column] = np.inf
        return class_gate(cost, tracks, detections)

    def _reidentify(self, detection):
        """화면 밖으로 나갔던 사람과 외형이 비슷하면 이전 ID를 돌려줍니다."""
        if detection.feature is None or not self.gallery:
            return None
        best, best_cost = None, self.config.reid_thresh
        for old in self.gallery:
            if old.cls != detection.cls or old.feature is None:
                continue
            old_height = old.box[3] - old.box[1]
            ratio = (detection._box[3] - detection._box[1]) / max(old_height, 1.0)
            cost = 1.0 - float(np.dot(old.feature, detection.feature))
            if 0.5 <= ratio <= 2.0 and cost < best_cost:
                best, best_cost = old, cost
        if best is not None:
            self.gallery.remove(best)
        return best

    # ------------------------------------------------------------------- update
    def update(self, boxes, scores, classes, features=None, frame_size=None):
        """한 프레임의 검출로 트랙을 갱신하고 현재 화면에 보이는 확정 트랙을 돌려줍니다.

        boxes: ``(N,4)`` x1y1x2y2, scores: ``(N,)``, classes: ``(N,)``,
        features: 길이 N의 외형 특징 목록 (없으면 ``None``), frame_size: ``(width, height)``.
        """
        cfg = self.config
        self.frame_id += 1
        frame_id = self.frame_id
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        classes = np.asarray(classes, dtype=int).reshape(-1)
        if features is None:
            features = [None] * len(boxes)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = area >= cfg.min_box_area
        high = keep & (scores >= cfg.high_thresh)
        low = keep & (scores > cfg.low_thresh) & ~high

        det_high = [STrack(boxes[i], scores[i], classes[i], features[i]) for i in np.flatnonzero(high)]
        det_low = [STrack(boxes[i], scores[i], classes[i]) for i in np.flatnonzero(low)]
        if len(det_high) > 1:  # 다른 사람과 많이 겹친 박스의 외형은 섞여 있어 특징 갱신에 쓰지 않습니다.
            overlap = iou_matrix([d._box for d in det_high], [d._box for d in det_high])
            np.fill_diagonal(overlap, 0)
            for detection, value in zip(det_high, overlap.max(axis=1)):
                detection.clean_feature = value < 0.3

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        pool = confirmed + self.lost
        for track in pool:
            track.predict(self.kalman, damp=cfg.damp_lost_velocity, freeze=cfg.freeze_out_of_frame)

        activated, refound, lost_now, removed_now = [], [], [], []

        def accept(track, detection):
            if track.state == TrackState.TRACKED:
                track.update(self.kalman, detection, frame_id, cfg.feature_momentum)
                activated.append(track)
            else:
                track.update(self.kalman, detection, frame_id, cfg.feature_momentum,
                             status=STATUS_REAPPEARED)
                refound.append(track)

        # 1차 연결: 높은 신뢰도 검출 ↔ 추적 중 + 잃어버린 트랙
        matches, unmatched_tracks, unmatched_high = linear_assignment(
            self._fused_cost(pool, det_high), cfg.match_thresh)
        for row, column in matches:
            accept(pool[row], det_high[column])

        # 2차 연결: 남은 추적 중 트랙 ↔ 낮은 신뢰도 검출 (가려져 점수가 떨어진 사람)
        remaining = [pool[i] for i in unmatched_tracks if pool[i].state == TrackState.TRACKED]
        cost = class_gate(iou_distance(remaining, det_low), remaining, det_low)
        matches, unmatched_remaining, _ = linear_assignment(cost, cfg.second_match_thresh)
        for row, column in matches:
            accept(remaining[row], det_low[column])
        for index in unmatched_remaining:
            track = remaining[index]
            track.mark_lost(frame_id, frame_size)
            lost_now.append(track)

        # 3차 연결(개선): 오래 가려졌던 트랙 ↔ 남은 높은 신뢰도 검출, 외형 기준
        detections = [det_high[i] for i in unmatched_high]
        if cfg.recover_lost:
            waiting = [pool[i] for i in unmatched_tracks if pool[i].state == TrackState.LOST
                       and pool[i] not in lost_now]
            matches, _, unmatched_det = linear_assignment(
                self._recover_cost(waiting, detections), cfg.recover_thresh)
            for row, column in matches:
                accept(waiting[row], detections[column])
            detections = [detections[i] for i in unmatched_det]

        # 아직 확정되지 않은 트랙(직전 프레임에 생긴 트랙) ↔ 남은 검출
        cost = class_gate(fuse_score(iou_distance(unconfirmed, detections), detections),
                          unconfirmed, detections)
        matches, unmatched_unconfirmed, unmatched_det = linear_assignment(
            cost, cfg.unconfirmed_match_thresh)
        for row, column in matches:
            unconfirmed[row].update(self.kalman, detections[column], frame_id, cfg.feature_momentum)
            activated.append(unconfirmed[row])
        for index in unmatched_unconfirmed:
            unconfirmed[index].state = TrackState.REMOVED
            removed_now.append(unconfirmed[index])

        # 새 트랙 (개선판은 화면 밖으로 나갔던 사람인지 먼저 확인)
        for index in unmatched_det:
            detection = detections[index]
            if detection.score < cfg.new_track_thresh:
                continue
            old = self._reidentify(detection) if self.reid_time else None
            if old is not None:
                old.mean, old.covariance = self.kalman.initiate(_to_xyah(detection._box))
                old.state = TrackState.LOST  # accept()에서 재활성화합니다.
                old.update(self.kalman, detection, frame_id, cfg.feature_momentum,
                           status=STATUS_REENTERED)
                refound.append(old)
                continue
            detection.activate(self.kalman, frame_id, self.next_id)
            self.next_id += 1
            activated.append(detection)

        # 오래 잃어버린 트랙 삭제 (개선판은 외형을 갤러리에 보관)
        for track in self.lost + lost_now:
            if track.state == TrackState.LOST and frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.REMOVED
                removed_now.append(track)
                if self.reid_time and track.feature is not None and track.is_activated:
                    track.removed_frame = frame_id
                    self.gallery.append(track)
        self.gallery = [t for t in self.gallery if frame_id - t.removed_frame <= self.reid_time]

        tracked = [t for t in self.tracked if t.state == TrackState.TRACKED]
        tracked = _unique(tracked + activated + refound)
        lost = _unique([t for t in self.lost + lost_now if t.state == TrackState.LOST])
        self.tracked, self.lost = _remove_duplicates(tracked, lost)
        return [t for t in self.tracked if t.is_activated]

    def lost_tracks(self):
        """가림·화면 이탈 상태로 기다리는 트랙 (GUI 표시용)."""
        return [t for t in self.lost if t.is_activated]


def _unique(tracks):
    seen, result = set(), []
    for track in tracks:
        if id(track) not in seen:
            seen.add(id(track))
            result.append(track)
    return result


def _remove_duplicates(tracked, lost):
    """추적 중 트랙과 거의 같은 위치의 Lost 트랙은 더 짧게 산 쪽을 지웁니다."""
    if not tracked or not lost:
        return tracked, lost
    overlap = iou_matrix([t.box for t in tracked], [t.box for t in lost])
    drop_tracked, drop_lost = set(), set()
    for row, column in zip(*np.nonzero(overlap > 0.85)):
        a, b = tracked[row], lost[column]
        if a.cls != b.cls:
            continue
        if a.frame_id - a.start_frame > b.frame_id - b.start_frame:
            drop_lost.add(column)
        else:
            drop_tracked.add(row)
    return ([t for i, t in enumerate(tracked) if i not in drop_tracked],
            [t for i, t in enumerate(lost) if i not in drop_lost])
