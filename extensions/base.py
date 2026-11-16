"""확장 기능 공통 틀.

파이프라인은 매 프레임 기본 추적(사람·동물)을 마친 뒤, 켜진 확장 기능마다 ``process(frame, frame_index)``를 부릅니다.
확장 기능은 화면에 그릴 :class:`pipeline.TrackRow` 목록을 돌려주기만 하면 됩니다.

* ``row.source``에는 확장 기능의 ``key``를 넣습니다. 화면·결과 파일에서 기본 추적과 구분하는 데 씁니다.
  (ID는 확장 기능마다 따로 1부터 시작하므로 ``source``와 함께 봐야 같은 객체입니다.)
* ``row.label``에는 화면에 함께 보일 글자(예: 인식한 한글 문장)를 넣습니다. 없으면 빈 문자열.
* ``frame``은 원본입니다 (얼굴 모자이크 전). 확장 기능이 ``frame``을 직접 고치면 안 됩니다.

대부분의 확장 기능은 "전용 검출기 + ByteTrack 추적 + (선택) 트랙별 후처리" 구조이므로
:class:`DetectTrackExtension`을 상속하면 검출·추적·이동 정보 계산을 그대로 쓸 수 있습니다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


class Extension:
    key = ""            # 영문 식별자. extensions/__init__.py의 REGISTRY 키와 같아야 합니다.
    label = ""          # GUI 체크박스에 보일 이름
    tag = ""            # 화면에서 ID 앞에 붙일 1~2글자 (예: "글" → "글3")
    description = ""

    def __init__(self, device="cpu", fps=30.0):
        self.device = device
        self.fps = fps

    @classmethod
    def check(cls):
        """사용할 수 없으면 이유(문자열), 사용할 수 있으면 ``None``. 가중치·패키지가 있는지 확인합니다."""
        return None

    def process(self, frame, frame_index):
        """한 프레임을 처리해 화면에 그릴 ``TrackRow`` 목록을 돌려줍니다."""
        raise NotImplementedError

    def close(self):
        """영상이 끝나거나 중지될 때 정리할 것이 있으면 여기서 합니다."""


class DetectTrackExtension(Extension):
    """전용 YOLO 검출기 + ByteTrack + 이동 정보. 하위 클래스는 아래 클래스 변수만 정하면 됩니다.

    * ``weights``: ``models/`` 아래 가중치 파일 이름
    * ``classes``: 추적할 클래스 이름 목록 (``None``이면 가중치의 모든 클래스)
    * ``tracker``: ``"baseline"``/``"improved"`` 또는 ``TrackerConfig`` 필드를 덮어쓸 dict
    * ``imgsz``: 검출 입력 크기
    * ``describe(frame, row)``: 트랙별 글자(``row.label``)를 정하고 싶으면 다시 정의합니다 (예: OCR).
    """
    weights = ""
    classes = None
    tracker = "baseline"
    imgsz = 640

    @classmethod
    def weights_path(cls):
        return MODEL_DIR / cls.weights

    @classmethod
    def check(cls):
        if not cls.weights_path().is_file():
            return f"가중치가 없습니다: models/{cls.weights or '(미정)'}"
        return None

    def __init__(self, device="cpu", fps=30.0):
        super().__init__(device, fps)
        from pipeline import Detector
        from tracker import ByteTracker, MotionRecorder, TrackerConfig

        reason = self.check()
        if reason:
            raise RuntimeError(f"{self.label}: {reason}")
        if self.classes is None:
            from ultralytics import YOLO

            classes = list(YOLO(str(self.weights_path())).names.values())
        else:
            classes = list(self.classes)
        self.detector = Detector(str(self.weights_path()), classes, self.imgsz, device)
        if isinstance(self.tracker, dict):
            base = TrackerConfig.preset(self.tracker.get("preset", "baseline"))
            for name, value in self.tracker.items():
                if name != "preset":
                    setattr(base, name, value)
            self.tracker_config = base
        else:
            self.tracker_config = TrackerConfig.preset(self.tracker)
        self.byte_tracker = ByteTracker(self.tracker_config, fps)
        self.motion = MotionRecorder(fps)

    def describe(self, frame, row):
        """트랙 하나에 붙일 글자. 기본은 없음."""
        return ""

    def process(self, frame, frame_index):
        from pipeline import TrackRow, needs_features
        from tracker import extract_features

        boxes, scores, classes = self.detector(frame, self.tracker_config.low_thresh)
        features = None
        if needs_features(self.tracker_config):
            features = extract_features(frame, boxes)
        height, width = frame.shape[:2]
        rows = []
        for track in self.byte_tracker.update(boxes, scores, classes, features, (width, height)):
            box = tuple(float(v) for v in track.box)
            state = self.motion.update(track.track_id, box, frame_index)
            row = TrackRow(track.track_id, self.detector.names[track.cls], box, track.score, track.status,
                           state.direction, state.distance, state.speed, tuple(state.trail),
                           source=self.key)
            row.label = self.describe(frame, row) or ""
            rows.append(row)
        return rows
