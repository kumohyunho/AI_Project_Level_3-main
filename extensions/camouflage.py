"""보호색 동물 전용 검출기와 ByteTrack. 학습 방법: docs/CAMOUFLAGE.md.

색 히스토그램은 배경과 비슷한 동물의 재식별을 오히려 방해할 수 있어 기본값은
위치 기반 ByteTrack입니다. 아래 임계값은 최적화 결과가 아닌 시작값입니다.
"""
import hashlib
import json
import math
from pathlib import Path

from tracker import TrackerConfig

from .base import DetectTrackExtension

CLASS_NAME = "camouflaged_animal"


def validate_nms(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("NMS IoU는 0과 1 사이의 유한한 수여야 합니다.")
    return float(value)


def read_inference_config(weights):
    """모델 옆의 선택적 실행 설정. 학습 메타데이터와 분리해 원본을 보존합니다."""
    weights = Path(weights)
    path = weights.with_suffix(".inference.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("extension") != "camouflage":
        raise ValueError("보호색 실행 설정 형식이 아닙니다.")
    if data.get("weights_sha256") != hashlib.sha256(weights.read_bytes()).hexdigest():
        raise ValueError("실행 설정과 모델 가중치의 해시가 다릅니다.")
    if data.get("max_det") != 300:
        raise ValueError("현재 보호색 실행 설정의 max_det은 300이어야 합니다.")
    return dict(nms_iou=validate_nms(data.get("nms_iou")), max_det=300)


class CamouflageDetector:
    """기존 로드 모델을 재사용하며 NMS를 YOLO predict에 직접 전달합니다."""
    def __init__(self, detector, nms_iou):
        self.base = detector
        self.names = detector.names
        self.nms_iou = validate_nms(nms_iou)

    def __call__(self, frame, conf):
        base = self.base
        result = base.model.predict(frame, conf=conf, iou=self.nms_iou, max_det=300,
                                    classes=base.class_ids, imgsz=base.imgsz,
                                    device=base.device, verbose=False)[0].boxes
        return (result.xyxy.cpu().numpy(), result.conf.cpu().numpy(),
                result.cls.cpu().numpy().astype(int))


def default_tracker():
    config = TrackerConfig.preset("baseline")
    config.high_thresh = 0.4
    config.low_thresh = 0.05
    config.new_track_thresh = 0.5
    config.track_buffer = 60
    return config


def validate_tracker(data):
    config = TrackerConfig.from_dict(data)
    thresholds = (config.low_thresh, config.high_thresh, config.new_track_thresh,
                  config.match_thresh, config.second_match_thresh, config.unconfirmed_match_thresh)
    if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in thresholds):
        raise ValueError("추적 임계값은 0~1이어야 합니다.")
    if not config.low_thresh < config.high_thresh <= config.new_track_thresh:
        raise ValueError("low_thresh < high_thresh <= new_track_thresh이어야 합니다.")
    if type(config.track_buffer) is not int or config.track_buffer < 1:
        raise ValueError("track_buffer는 양의 정수여야 합니다.")
    return config


def read_tracker_config(path, weights=None):
    """튜닝 결과 또는 순수 TrackerConfig JSON을 공통으로 읽습니다."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if weights is not None and "weights_sha256" in data:
        actual = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        if data["weights_sha256"] != actual:
            raise ValueError("튜닝에 사용한 가중치와 현재 가중치가 다릅니다.")
    return validate_tracker(data.get("tracker", data)).to_dict()


class CamouflageExtension(DetectTrackExtension):
    key = "camouflage"
    label = "보호색 동물"
    tag = "보"
    description = "전용 YOLO로 보호색 동물을 찾고 ByteTrack으로 ID와 이동 경로를 유지합니다."
    weights = "camouflage.pt"
    classes = [CLASS_NAME]
    tracker = default_tracker().to_dict()

    @classmethod
    def metadata(cls):
        data = json.loads(cls.weights_path().with_suffix(".json").read_text(encoding="utf-8"))
        if data.get("extension") != cls.key or data.get("classes") != [CLASS_NAME]:
            raise ValueError("보호색 동물 단일 클래스 학습 정보가 아닙니다.")
        validate_tracker(data.get("tracker", default_tracker().to_dict()))
        return data

    @classmethod
    def check(cls):
        reason = super().check()
        if reason:
            return reason + " (train_camouflage.py --install로 학습·설치하세요.)"
        try:
            cls.metadata()
            read_inference_config(cls.weights_path())
        except (OSError, ValueError, TypeError, AttributeError) as error:
            return f"학습/실행 설정 JSON을 확인하세요: {error}"
        return None

    def __init__(self, device="cpu", fps=30.0, nms_iou=None):
        reason = self.check()
        if reason:
            raise RuntimeError(f"{self.label}: {reason}")
        self.tracker = validate_tracker(self.metadata().get("tracker", self.tracker)).to_dict()
        configured = read_inference_config(self.weights_path())
        if nms_iou is not None:
            configured = dict(nms_iou=validate_nms(nms_iou), max_det=300)
        super().__init__(device, fps)
        self.inference_config = configured or dict(nms_iou=0.7, max_det=300)
        if configured is not None:
            self.detector = CamouflageDetector(self.detector, configured["nms_iou"])

    def describe(self, frame, row):
        return "보호색 동물"
