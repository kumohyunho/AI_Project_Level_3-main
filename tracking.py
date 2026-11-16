"""영상·카메라를 읽어 파이프라인으로 추적하고 Qt 화면에 프레임별 결과를 전달합니다 (요구 사항 8).

``replay``를 주면 모델을 실행하지 않고 저장된 추적 결과 파일을 영상 위에 다시 그립니다 (요구 사항 10).
"""
from math import isfinite
from threading import Condition, Event

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from pipeline import TrackRow, TrackingPipeline
from tracker import STATUS_EXITED, STATUS_OCCLUDED, MotionRecorder


class TrackingThread(QThread):
    metadata = Signal(int, float, bool)            # 길이(ms), FPS, 카메라 여부
    frame_ready = Signal(QImage, list, list, int, int, dict)  # 이미지, 트랙, 대기 트랙, 위치(ms), 프레임 번호, 시간
    failed = Signal(str)

    def __init__(self, source, config=None, replay=None, parent=None):
        super().__init__(parent)
        self.source = source            # 파일 경로 또는 카메라 번호(int)
        self.config = config
        self.replay = replay
        self.control = Condition()
        self.paused = False
        self.cancelled = False
        self.displayed = Event()
        self.tracker_config = None

    def set_paused(self, paused):
        with self.control:
            self.paused = paused
            self.control.notify_all()

    def cancel(self):
        with self.control:
            self.cancelled = True
            self.control.notify_all()
        self.displayed.set()

    def run(self):
        try:
            self.track_video()
        except Exception as error:
            self.failed.emit(str(error))

    def _open(self):
        import cv2

        camera = isinstance(self.source, int)
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise ValueError(f"카메라 {self.source}번을 열 수 없습니다." if camera else "영상을 열 수 없습니다.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if isfinite(fps) and 0 < fps < 240 else 30.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = 0 if camera or not isfinite(frame_count) else round(frame_count * 1000 / fps)
        return capture, fps, max(0, duration), camera

    def track_video(self):
        capture, fps, duration, camera = self._open()
        try:
            ok, frame = capture.read()
            if not ok:
                raise ValueError("영상 프레임을 읽을 수 없습니다.")
            self.metadata.emit(duration, fps, camera)
            pipeline = None
            motion = MotionRecorder(fps)
            if self.replay is None:
                pipeline = TrackingPipeline(self.config, fps)
                self.tracker_config = pipeline.tracker_config
            index = 0
            while ok:
                with self.control:
                    self.control.wait_for(lambda: not self.paused or self.cancelled)
                    if self.cancelled:
                        return
                index += 1
                if pipeline is not None:
                    rows, waiting, timings = pipeline.process(frame)
                else:
                    rows, waiting = self._replay_rows(index, motion)
                    timings = {}
                height, width, _ = frame.shape
                image = QImage(frame.data, width, height, frame.strides[0], QImage.Format_BGR888).copy()
                # 화면이 그릴 때까지 기다려 프레임이 신호 큐에 쌓이지 않게 합니다.
                self.displayed.clear()
                with self.control:
                    if self.cancelled:
                        return
                self.frame_ready.emit(image, rows, waiting, round((index - 1) * 1000 / fps), index, timings)
                self.displayed.wait()
                ok, frame = capture.read()
        finally:
            capture.release()

    def _replay_rows(self, index, motion):
        """저장된 행을 화면용 트랙으로 바꿉니다. 가림·화면 이탈 상태는 대기 목록으로 분리합니다."""
        rows, waiting = [], []
        for item in self.replay.get(index, []):
            if item["status"] in (STATUS_OCCLUDED, STATUS_EXITED):
                waiting.append(TrackRow(item["id"], item["class"], item["box"], item["score"],
                                        item["status"], item["direction"], item["distance"], 0.0, ()))
                continue
            state = motion.update(item["id"], item["box"], index)
            rows.append(TrackRow(item["id"], item["class"], item["box"], item["score"], item["status"],
                                 item["direction"], item["distance"], item["speed"], tuple(state.trail)))
        return rows, waiting
