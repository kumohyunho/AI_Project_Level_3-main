"""확장 기능: 떠다니는 한글 자음·모음 검출·추적 (담당: Pdint / Claude).  설계: ``docs/HANGUL.md``

자음·모음 **하나하나가 추적 대상**입니다. 검출기가 40개 자모를 바로 분류하므로 OCR은 쓰지 않습니다.

1. 검출: 합성 데이터(``hangul_dataset.py``)로 YOLO11n Fine-tuning → ``models/hangul_jamo.pt`` (40클래스)
2. 추적: 자모는 회전하면 다른 글자로 보이므로(ㅏ↔ㅓ, ㅗ↔ㅜ) 클래스와 무관하게 위치·움직임으로 연결
3. 분류: 트랙별로 여러 프레임의 검출 클래스를 모아 다수결로 자모를 정해 표시

``models/hangul_jamo.pt``가 있으면 GUI의 "한글 자모" 체크박스가 활성화됩니다.
"""
from collections import defaultdict

import numpy as np

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


class JamoVotes:
    """트랙별 자모 다수결. 검출 신뢰도를 표로 더해, 지금까지 가장 많이 지지받은 자모를 고릅니다.

    회전·가림 때문에 몇 프레임 다른 자모로 검출되어도 화면에 보이는 자모는 바뀌지 않습니다.
    """

    def __init__(self):
        self.scores = defaultdict(lambda: defaultdict(float))
        self.frames = defaultdict(int)       # 트랙별 투표한 프레임 수
        self.disagree = defaultdict(int)     # 그 프레임의 검출이 다수결 결과와 달랐던 횟수

    def add(self, track_id, class_name, score):
        self.scores[track_id][class_name] += float(score)
        self.frames[track_id] += 1
        if class_name != self.best(track_id):
            self.disagree[track_id] += 1

    def best(self, track_id):
        votes = self.scores.get(track_id)
        return max(votes, key=votes.get) if votes else None

    def confidence(self, track_id):
        """다수결 1위가 전체 표에서 차지하는 비율 (0~1)."""
        votes = self.scores.get(track_id)
        return max(votes.values()) / sum(votes.values()) if votes else 0.0


class HangulJamoExtension(DetectTrackExtension):
    key = "hangul"
    label = "한글 자모"
    tag = "자"
    description = "떠다니는 한글 자음·모음을 하나씩 검출·추적하고 어떤 자모인지 표시합니다."
    weights = "hangul_jamo.pt"
    classes = CLASS_NAMES      # 가중치에 40개 자모 클래스가 모두 있어야 합니다.
    # 자모는 색이 서로 달라 외형(색 히스토그램) 연결이 도움이 됩니다. 평가로 기본판과 비교합니다 (docs/HANGUL.md).
    tracker = "improved"
    match_iou = 0.3            # 트랙과 이번 프레임 검출을 같은 것으로 보는 최소 IoU (투표용)

    def __init__(self, device="cpu", fps=30.0):
        super().__init__(device, fps)
        self.votes = JamoVotes()

    def detect(self, frame):
        """(boxes, scores, class_ids). 같은 자모가 ㅏ·ㅓ 두 클래스로 겹쳐 검출되지 않도록 클래스와 무관하게 NMS합니다."""
        result = self.detector.model.predict(
            frame, conf=self.tracker_config.low_thresh, classes=self.detector.class_ids, imgsz=self.imgsz,
            device=self.device, agnostic_nms=True, verbose=False)[0].boxes
        return (result.xyxy.cpu().numpy(), result.conf.cpu().numpy(), result.cls.cpu().numpy().astype(int))

    def class_name(self, class_id):
        return self.detector.names[int(class_id)]

    def reset(self, tracker=None):
        """추적 상태와 투표를 처음으로 되돌립니다. ``tracker``를 주면 추적기 설정도 바꿉니다.

        ``tracker``: ``"baseline"``, ``"improved"``, ``"baseline+appearance"``, ``configs/<이름>.json``의 이름,
        또는 ``TrackerConfig``. 같은 검출 위에서 추적기 설정만 바꿔 비교할 때 씁니다 (``evaluate_hangul.py``).
        """
        from pipeline import tracker_config
        from tracker import ByteTracker, MotionRecorder, TrackerConfig

        if tracker is not None:
            self.tracker_config = tracker if isinstance(tracker, TrackerConfig) else tracker_config(tracker)
        self.byte_tracker = ByteTracker(self.tracker_config, self.fps)
        self.motion = MotionRecorder(self.fps)
        self.votes = JamoVotes()

    def track(self, boxes, scores, class_ids, features, frame_size):
        """한 프레임의 검출로 추적·투표합니다.

        반환: ``[(트랙, 다수결 자모 이름 또는 None, 이번 프레임에 검출된 자모 이름 또는 None), ...]``
        """
        from tracker.matching import iou_matrix

        # 추적기에는 모든 검출을 같은 클래스(0)로 넘깁니다. 클래스가 프레임마다 바뀌어도 ID가 끊기지 않습니다.
        tracks = self.byte_tracker.update(boxes, scores, np.zeros(len(boxes), dtype=int), features, frame_size)
        observed = {}
        if tracks and len(boxes):
            overlap = iou_matrix([t.box for t in tracks], boxes)
            # 높은 신뢰도 검출만 투표합니다 (2차 연결용 낮은 신뢰도 검출은 분류가 불안정).
            overlap[:, np.asarray(scores) < self.tracker_config.high_thresh] = 0.0
            for row, track in enumerate(tracks):
                column = int(overlap[row].argmax())
                if overlap[row, column] >= self.match_iou:
                    observed[track.track_id] = self.class_name(class_ids[column])
                    self.votes.add(track.track_id, observed[track.track_id], scores[column])
        return [(t, self.votes.best(t.track_id), observed.get(t.track_id)) for t in tracks]

    def process(self, frame, frame_index):
        from pipeline import TrackRow, needs_features
        from tracker import extract_features

        boxes, scores, class_ids = self.detect(frame)
        features = extract_features(frame, boxes) if needs_features(self.tracker_config) and len(boxes) else None
        height, width = frame.shape[:2]
        rows = []
        for track, name, _ in self.track(boxes, scores, class_ids, features, (width, height)):
            if name is None:   # 아직 높은 신뢰도 검출과 만나지 못한 트랙
                continue
            box = tuple(float(v) for v in track.box)
            state = self.motion.update(track.track_id, box, frame_index)
            rows.append(TrackRow(track.track_id, KIND.get(name, name), box, track.score, track.status,
                                 state.direction, state.distance, state.speed, tuple(state.trail),
                                 source=self.key, label=CHAR.get(name, name)))
        return sorted(rows, key=lambda r: r.track_id)
