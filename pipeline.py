"""검출(YOLO) → 외형 특징 → 추적(ByteTracker) → 얼굴 모자이크를 하나로 묶은 처리 파이프라인.

GUI, MOT 평가, 파라미터 탐색, 속도 측정이 모두 이 파일을 통해 같은 설정으로 동작합니다.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from tracker import ByteTracker, MotionRecorder, TrackerConfig, extract_features

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "configs"
DEFAULT_CLASSES = ["person"]
TRACKABLE_CLASSES = ["person", "cat", "dog", "bird"]   # COCO에서 고른 추적 대상


def default_weights():
    for candidate in (MODEL_DIR / "person.pt", MODEL_DIR / "coco.pt", ROOT / "yolo11n.pt"):
        if candidate.is_file():
            return str(candidate)
    return "yolo11n.pt"


def available_weights():
    """GUI 모델 선택 목록: 사람·동물 검출용 가중치만 (얼굴 모델 제외)."""
    found = []
    for folder in (MODEL_DIR, ROOT / "experiments", ROOT):
        for path in sorted(folder.glob("*.pt")):
            if path.stem != "face" and path not in found:
                found.append(path)
    return [str(path) for path in found]


def tracker_config(name_or_path):
    """``baseline`` / ``improved`` 프리셋 또는 ``configs/*.json`` 파일에서 추적기 설정을 읽습니다."""
    import json

    path = Path(name_or_path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return TrackerConfig.from_dict(data.get("tracker", data))
    tuned = CONFIG_DIR / f"{name_or_path}.json"
    if tuned.is_file():
        return tracker_config(tuned)
    if "+" in str(name_or_path):  # 예: baseline+appearance → 개선 항목 하나만 켠 ablation 설정
        base, *options = str(name_or_path).split("+")
        config = TrackerConfig.preset(base)
        for option in options:
            if option not in ABLATION_OPTIONS:
                raise ValueError(f"알 수 없는 개선 항목: {option} (가능: {', '.join(ABLATION_OPTIONS)})")
            setattr(config, option, ABLATION_OPTIONS[option])
        return config
    return TrackerConfig.preset(name_or_path)


# 개선 항목을 하나씩 켜서 효과를 따로 측정하기 위한 값 (run_mot.py --trackers baseline+appearance ...)
ABLATION_OPTIONS = {"appearance": True, "recover_lost": True, "reid_buffer": 150,
                    "freeze_out_of_frame": True, "damp_lost_velocity": True}


def tracker_choices():
    """GUI 추적기 목록: 프리셋 + 튜닝 결과 파일."""
    names = ["baseline", "improved"]
    names += [p.stem for p in sorted(CONFIG_DIR.glob("*.json")) if p.stem not in names]
    return names


@dataclass
class PipelineConfig:
    weights: str = field(default_factory=default_weights)
    classes: list = field(default_factory=lambda: list(DEFAULT_CLASSES))
    tracker: str = "improved"
    imgsz: int = 640
    device: str = "cpu"
    mosaic: bool = True
    face_weights: str = str(MODEL_DIR / "face.pt")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class TrackRow:
    track_id: int
    cls_name: str
    box: tuple
    score: float
    status: str
    direction: str = "—"
    distance: float = 0.0
    speed: float = 0.0
    trail: tuple = ()


class Detector:
    def __init__(self, weights, classes, imgsz=640, device="cpu"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        names = {name.casefold(): int(i) for i, name in self.model.names.items()}
        missing = [c for c in classes if c.casefold() not in names]
        if missing:
            raise ValueError(f"가중치 {Path(weights).name}에 없는 클래스: {', '.join(missing)}")
        self.class_ids = [names[c.casefold()] for c in classes]
        self.names = self.model.names
        self.imgsz, self.device = imgsz, device

    def __call__(self, frame, conf):
        result = self.model.predict(frame, conf=conf, classes=self.class_ids, imgsz=self.imgsz,
                                    device=self.device, verbose=False)[0].boxes
        return (result.xyxy.cpu().numpy(), result.conf.cpu().numpy(),
                result.cls.cpu().numpy().astype(int))


class FaceMosaic:
    """확장 기능: WIDER FACE로 학습한 얼굴 검출기로 찾은 얼굴을 모자이크합니다."""

    def __init__(self, weights, device="cpu"):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        if set(self.model.names.values()) != {"face"}:
            raise ValueError(f"{Path(weights).name}는 face 단일 클래스 가중치여야 합니다.")
        self.device = device

    def __call__(self, frame, block=14):
        import cv2

        boxes = self.model.predict(frame, imgsz=640, conf=0.2, device=self.device,
                                   verbose=False)[0].boxes
        height, width = frame.shape[:2]
        for x1, y1, x2, y2 in boxes.xyxy.int().cpu().tolist():
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            region = frame[y1:y2, x1:x2]
            small = cv2.resize(region, (max(1, (x2 - x1) // block), max(1, (y2 - y1) // block)),
                               interpolation=cv2.INTER_AREA)
            region[:] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        return len(boxes)


def needs_features(config):
    return config.appearance or config.recover_lost or config.reid_buffer > 0


class TrackingPipeline:
    def __init__(self, config, fps=30.0):
        self.config = config
        self.tracker_config = tracker_config(config.tracker)
        self.detector = Detector(config.weights, config.classes, config.imgsz, config.device)
        self.mosaic = None
        if config.mosaic and Path(config.face_weights).is_file():
            self.mosaic = FaceMosaic(config.face_weights, config.device)
        self.tracker = ByteTracker(self.tracker_config, fps)
        self.motion = MotionRecorder(fps)
        self.frame_index = 0

    def process(self, frame):
        """프레임 하나를 처리합니다. 모자이크는 ``frame``에 직접 적용됩니다.

        반환: (보이는 트랙 목록, 가림·이탈 트랙 목록, 단계별 처리 시간(ms) 딕셔너리)
        """
        timings = {}
        started = perf_counter()
        boxes, scores, classes = self.detector(frame, self.tracker_config.low_thresh)
        timings["detect"] = (perf_counter() - started) * 1000

        started = perf_counter()
        features = None
        if needs_features(self.tracker_config):
            high = scores >= self.tracker_config.high_thresh
            features = [None] * len(boxes)
            for index, feature in zip(np.flatnonzero(high), extract_features(frame, boxes[high])):
                features[index] = feature
        height, width = frame.shape[:2]
        tracks = self.tracker.update(boxes, scores, classes, features, (width, height))
        timings["track"] = (perf_counter() - started) * 1000

        timings["mosaic"] = 0.0
        if self.mosaic is not None:
            started = perf_counter()
            self.mosaic(frame)
            timings["mosaic"] = (perf_counter() - started) * 1000

        self.frame_index += 1
        rows = []
        for track in tracks:
            box = tuple(float(v) for v in track.box)
            state = self.motion.update(track.track_id, box, self.frame_index)
            rows.append(TrackRow(track.track_id, self.detector.names[track.cls], box, track.score,
                                 track.status, state.direction, state.distance, state.speed,
                                 tuple(state.trail)))
        waiting = []
        for track in self.tracker.lost_tracks():
            state = self.motion.states.get(track.track_id)
            waiting.append(TrackRow(track.track_id, self.detector.names[track.cls],
                                    tuple(float(v) for v in track.box), track.score, track.status,
                                    state.direction if state else "—",
                                    state.distance if state else 0.0, 0.0, ()))
        timings["total"] = timings["detect"] + timings["track"] + timings["mosaic"]
        return sorted(rows, key=lambda r: r.track_id), sorted(waiting, key=lambda r: r.track_id), timings
