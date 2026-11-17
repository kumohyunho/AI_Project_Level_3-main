"""보호색 전용 화면: 데이터 정답 미리보기·모델 정보·원본/추적 영상·학습 비교 진입."""
import codecs
import hashlib
import json
import math
import sys
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
                               QLabel, QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy,
                               QSlider, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from extensions.camouflage import CLASS_NAME, default_tracker, read_inference_config, validate_tracker

ROOT = Path(__file__).resolve().parent


def dataset_records(path):
    path = Path(path).resolve()
    root = path if path.is_dir() else path.parent
    manifest = json.loads((root/"splits.json").read_text(encoding="utf-8"))
    if manifest.get("classes") != [CLASS_NAME]:
        raise ValueError("보호색 단일 클래스 데이터의 splits.json이 필요합니다.")
    records, seen = [], set()
    for row in manifest["records"]:
        relative = Path(row["image"])
        split = row["split"]
        if (split not in ("train", "val", "test") or relative.is_absolute() or len(relative.parts) != 3
                or relative.parts[:2] != ("images", split) or type(row["boxes"]) is not int or row["boxes"] < 0):
            raise ValueError("이미지 경로·분할·정답 개수 형식이 올바르지 않습니다.")
        image = (root/relative).resolve()
        if not image.is_relative_to(root) or image in seen:
            raise ValueError("데이터 폴더 밖 경로 또는 중복 이미지 경로입니다.")
        seen.add(image)
        records.append(row)
    if not records:
        raise ValueError("이미지 목록이 비어 있습니다.")
    for split in ("train", "val", "test"):
        if sum(row["split"] == split for row in records) != manifest["counts"][split]["images"]:
            raise ValueError(f"{split} 이미지 개수가 명세와 다릅니다.")
    return root, records


def ground_truth_preview(root, row):
    root = Path(root).resolve()
    relative = Path(row["image"])
    image = (root/relative).resolve()
    label = (root/"labels"/row["split"]/relative.with_suffix(".txt").name).resolve()
    if not image.is_relative_to(root) or not label.is_relative_to(root):
        raise ValueError("데이터 폴더 밖 경로입니다.")
    with Image.open(image) as source:
        preview = source.convert("RGB")
    width, height = preview.size
    lines = label.read_text(encoding="utf-8").splitlines()
    if len(lines) != row["boxes"]:
        raise ValueError("선택 이미지의 정답 개수가 명세와 다릅니다.")
    draw = ImageDraw.Draw(preview)
    for line in lines:
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError("선택 이미지의 YOLO 라벨 형식 오류입니다.")
        cls, x, y, w, h = values
        if (cls != 0 or not 0 < w <= 1 or not 0 < h <= 1 or
                min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6):
            raise ValueError("선택 이미지의 정답 클래스·좌표 범위 오류입니다.")
        box = (min(width-1, max(0, (x-w/2)*width)), min(height-1, max(0, (y-h/2)*height)),
               min(width-1, (x+w/2)*width), min(height-1, (y+h/2)*height))
        draw.rectangle(box, outline="#00ff70", width=max(2, width//400))
        draw.text((box[0], max(0, box[1]-12)), "GT", fill="#00ff70")
    preview.thumbnail((1000, 700))
    return preview


def model_information(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"모델이 없습니다: {path}")
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if data.get("extension") != "camouflage" or data.get("classes") != [CLASS_NAME]:
        raise ValueError("보호색 모델과 같은 학습의 JSON이 필요합니다.")
    validate_tracker(data.get("tracker", default_tracker().to_dict()))
    if "weights_sha256" in data and hashlib.sha256(path.read_bytes()).hexdigest() != data["weights_sha256"]:
        raise ValueError("모델 가중치와 학습 JSON의 SHA256이 다릅니다.")
    return data, read_inference_config(path) or dict(nms_iou=0.7, max_det=300)


class CamouflagePage(QWidget):
    processing_finished = Signal()

    def __init__(self, parent=None, on_learning=None):
        super().__init__(parent)
        self.dataset_root = self.model_path = self.video_path = self.result_video = None
        self.records, self.visible_records = [], []
        self.preview_image = None
        self.process = None
        self.diagnostic_root = None
        self.cancelled = self.close_pending = self.external_busy = False
        layout = QVBoxLayout(self)
        actions = QHBoxLayout()
        self.dataset_button = self.button(actions, "학습 데이터 열기", self.choose_dataset)
        self.model_button = self.button(actions, "보호색 모델 열기", self.choose_model)
        self.video_button = self.button(actions, "영상 열기", self.choose_video)
        self.learning_button = self.button(actions, "보호색 학습 비교", on_learning or self.open_learning)
        self.button(actions, "추적 결과 폴더 열기", self.open_output)
        self.diagnostic_button = self.button(actions, "진단 요약 열기", self.open_diagnostics)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addWidget(self.text_label("학습 데이터의 정답과 모델의 검출 결과를 구분해 확인하세요. 모델·원본 영상만으로 추적할 수 있습니다."))
        columns = QSplitter(Qt.Horizontal)
        left = QWidget()
        data_layout = QVBoxLayout(left)
        self.dataset_info = self.text_label("학습 데이터 미선택 — dataset.yaml 또는 splits.json을 선택하세요.")
        data_layout.addWidget(self.dataset_info)
        self.split = QComboBox()
        for name, value in (("학습 (train)", "train"), ("검증 (val)", "val"), ("테스트 (test)", "test")):
            self.split.addItem(name, value)
        self.split.currentIndexChanged.connect(self.show_records)
        data_layout.addWidget(self.split)
        self.images = QTableWidget(0, 3)
        self.images.setHorizontalHeaderLabels(["이미지", "출처", "정답 박스"])
        self.images.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.images.setSelectionBehavior(QTableWidget.SelectRows)
        self.images.setSelectionMode(QTableWidget.SingleSelection)
        self.images.setEditTriggers(QTableWidget.NoEditTriggers)
        self.images.setMaximumHeight(240)
        self.images.currentCellChanged.connect(self.show_preview)
        data_layout.addWidget(self.images)
        data_layout.addWidget(self.text_label("정답 박스 미리보기 — YOLO 예측·추적 ID가 아닙니다."))
        self.preview = self.text_label("이미지를 선택하면 정답 박스를 표시합니다.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(300, 200)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        data_layout.addWidget(self.preview, 1)
        self.preview_info = self.text_label("명세 목록과 선택한 파일만 확인합니다. 전체 데이터 검사는 별도로 필요합니다.")
        data_layout.addWidget(self.preview_info)
        columns.addWidget(left)
        right = QWidget()
        video_layout = QVBoxLayout(right)
        self.model_info = self.text_label("보호색 모델 미선택")
        video_layout.addWidget(self.model_info)
        self.video_info = self.text_label("원본 영상 미선택")
        video_layout.addWidget(self.video_info)
        self.screen = QVideoWidget()
        self.screen.setStyleSheet("background-color: #05090d;")
        self.screen.setMinimumSize(480, 270)
        video_layout.addWidget(self.screen, 1)
        self.player = QMediaPlayer(self)
        self.player.setVideoOutput(self.screen)
        self.player.errorOccurred.connect(lambda _, message: self.status.setText(f"영상 재생 오류: {message}"))
        controls = QHBoxLayout()
        self.raw_button = self.button(controls, "원본 재생", self.play_original)
        self.track_button = self.button(controls, "보호색 추적 실행", self.start_tracking)
        self.result_button = self.button(controls, "추적 결과 재생", self.play_result)
        self.diagnostic_video_button = self.button(controls, "진단 영상 재생", self.play_diagnostic)
        video_layout.addLayout(controls)
        playback = QHBoxLayout()
        self.button(playback, "재생 일시정지", self.player.pause)
        self.button(playback, "중지", self.stop_all)
        self.diagnostics = QCheckBox("진단 기록 저장")
        self.diagnostics.setChecked(True)
        playback.addWidget(self.diagnostics)
        self.device = QComboBox()
        self.device.addItem("CPU", "cpu")
        self.device.addItem("GPU 0 (CUDA 필요)", "0")
        playback.addWidget(self.device)
        self.clock = QLabel("00:00 / 00:00")
        playback.addWidget(self.clock)
        video_layout.addLayout(playback)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.player.durationChanged.connect(lambda duration: self.slider.setRange(0, duration))
        self.player.positionChanged.connect(self.position_changed)
        video_layout.addWidget(self.slider)
        columns.addWidget(right)
        columns.setStretchFactor(0, 2)
        columns.setStretchFactor(1, 3)
        layout.addWidget(columns, 1)
        self.status = self.text_label("대기 — 영상을 선택하면 원본 재생 또는 보호색 추적을 실행할 수 있습니다.")
        layout.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setMaximumHeight(95)
        layout.addWidget(self.log)
        self.set_busy(False)
        if (ROOT/"models/camouflage.pt").is_file():
            try:
                self.load_model(ROOT/"models/camouflage.pt")
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.model_info.setText(str(error))
        if (ROOT/"datasets/camouflage/splits.json").is_file():
            try:
                self.load_dataset(ROOT/"datasets/camouflage")
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.dataset_info.setText(str(error))

    @staticmethod
    def text_label(text):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.PlainText)
        return label

    @staticmethod
    def button(layout, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def choose_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "보호색 데이터 선택", str(ROOT), "데이터 명세 (dataset.yaml splits.json)")
        if path:
            self.checked(lambda: self.load_dataset(path))

    def load_dataset(self, path):
        root, records = dataset_records(path)
        self.dataset_root, self.records = root, records
        counts = " · ".join(f"{split} {sum(r['split'] == split for r in records)}장" for split in ("train", "val", "test"))
        self.dataset_info.setText(f"명세 목록: {counts}\n{root}")
        for index, split in enumerate(("train", "val", "test")):
            if (root/"images"/split).is_dir():
                self.split.setCurrentIndex(index)
                break
        self.show_records()

    def show_records(self):
        self.visible_records = [row for row in self.records if row["split"] == self.split.currentData()]
        self.images.blockSignals(True)
        self.images.setRowCount(len(self.visible_records))
        for index, row in enumerate(self.visible_records):
            for column, value in enumerate((Path(row["image"]).name, row.get("source", "—"), row["boxes"])):
                self.images.setItem(index, column, QTableWidgetItem(str(value)))
        self.preview_image = None
        self.preview.clear()
        if self.visible_records:
            self.images.setCurrentCell(0, 0)
        self.images.blockSignals(False)
        if self.visible_records:
            self.show_preview(0)
        else:
            self.preview.setText("이 분할에는 등록된 이미지가 없습니다.")
            self.preview_info.setText("선택한 분할의 이미지 목록이 비어 있습니다.")

    def show_preview(self, row, *unused):
        if not 0 <= row < len(self.visible_records):
            return
        record = self.visible_records[row]
        try:
            preview = ground_truth_preview(self.dataset_root, record)
            self.preview_image = QImage(preview.tobytes(), preview.width, preview.height,
                                        preview.width*3, QImage.Format_RGB888).copy()
            self.fit_preview()
            self.preview_info.setText(f"{record['image']} · 정답 박스 {record['boxes']}개\n실제 개체 ID·보호색 여부의 검증을 대신하지 않습니다.")
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.preview_image = None
            self.preview.clear()
            self.preview.setText("이미지·정답 파일을 확인하세요. 데이터는 코드에 포함되지 않습니다.")
            self.preview_info.setText(str(error))

    def fit_preview(self):
        if self.preview_image is not None:
            self.preview.setPixmap(QPixmap.fromImage(self.preview_image).scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_preview()

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "보호색 모델 선택", str(ROOT/"models"), "모델 (*.pt)")
        if path:
            self.checked(lambda: self.load_model(path))

    def load_model(self, path):
        data, inference = model_information(path)
        options = data.get("train_options", {})
        scores = data.get("scores", {}).get("val", {})
        score = scores.get("mAP50")
        text = f"{score*100:.2f}%" if type(score) in (int, float) and math.isfinite(score) else "미기록"
        self.model_path = Path(path).resolve()
        self.model_info.setText(f"모델: {self.model_path}\n학습 기록: 에폭 {options.get('epochs', '미기록')} · 초기 학습률 {options.get('lr0', '미기록')} · val mAP50 {text} (재측정 아님)\n실행 NMS IoU: {inference['nms_iou']}")
        self.result_video = None
        self.diagnostic_root = None
        self.player.stop()
        if self.video_path:
            self.player.setSource(QUrl.fromLocalFile(str(self.video_path)))
            self.video_info.setText(f"원본: {self.video_path}")
        self.set_busy(self.is_busy())

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "원본 영상 선택", "", "영상 (*.mp4 *.avi *.mov *.mkv *.webm)")
        if path:
            self.load_video(path)

    def load_video(self, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"영상이 없습니다: {path}")
        self.player.stop()
        self.video_path, self.result_video = path, None
        self.diagnostic_root = None
        self.video_info.setText(f"원본: {path}")
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.set_busy(self.is_busy())

    def play_original(self):
        if self.video_path:
            self.player.setSource(QUrl.fromLocalFile(str(self.video_path)))
            self.player.play()
            self.video_info.setText(f"원본 재생: {self.video_path.name}")

    def play_result(self):
        if self.result_video and self.result_video.is_file():
            self.player.setSource(QUrl.fromLocalFile(str(self.result_video)))
            self.player.play()
            self.video_info.setText(f"보호색 추적 결과: {self.result_video}")

    def is_busy(self):
        return self.process is not None and self.process.state() != QProcess.NotRunning

    def set_busy(self, active):
        for widget in (self.dataset_button, self.model_button, self.video_button, self.device, self.learning_button, self.diagnostics):
            widget.setEnabled(not active)
        self.learning_button.setProperty("available", not active)
        self.learning_button.setEnabled(not active and not self.external_busy)
        self.track_button.setEnabled(not active and not self.external_busy
                                     and self.video_path is not None and self.model_path is not None)
        self.raw_button.setEnabled(not active and self.video_path is not None)
        self.result_button.setEnabled(not active and self.result_video is not None)
        self.diagnostic_button.setEnabled(not active and self.diagnostic_root is not None)
        self.diagnostic_video_button.setEnabled(not active and self.diagnostic_root is not None)

    def set_external_busy(self, active):
        self.external_busy = active
        self.set_busy(self.is_busy())

    def start_tracking(self):
        if self.is_busy() or self.external_busy or not self.video_path or not self.model_path:
            return
        self.player.stop()
        self.result_video = None
        self.diagnostic_root = None
        self.current_output = ROOT/"runs/camouflage"/f"gui_{uuid4().hex[:8]}"
        self.cancelled = False
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.log.clear()
        self.status.setText("보호색 추적 처리 중 — 완료 후 결과 영상을 재생합니다. 학습은 수행하지 않습니다.")
        self.set_busy(True)
        process = QProcess(self)
        self.process = process
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUTF8", "1")
        process.setProcessEnvironment(environment)
        process.setWorkingDirectory(str(ROOT))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self.read_log)
        process.finished.connect(self.tracking_finished)
        process.errorOccurred.connect(self.process_error)
        process.start(sys.executable, ["-u", str(ROOT/"run_camouflage.py"), "--video", str(self.video_path),
                                      "--weights", str(self.model_path), "--device", self.device.currentData(),
                                      "--output", str(self.current_output),
                                      *(["--diagnostics"] if self.diagnostics.isChecked() else [])])

    def read_log(self):
        self.log.appendPlainText(self.decoder.decode(bytes(self.process.readAllStandardOutput())))

    def process_error(self, error):
        if error == QProcess.FailedToStart:
            self.set_busy(False)
            self.status.setText(f"실행 시작 실패: {self.process.errorString()}")
            self.processing_finished.emit()
            if self.close_pending:
                self.close()

    def tracking_finished(self, code, exit_status):
        self.read_log()
        result = self.current_output/"tracked.mp4"
        if self.cancelled:
            self.status.setText("중지됨 — 부분 영상은 완료된 추적 결과로 표시하지 않습니다.")
        elif code == 0 and all((self.current_output/name).is_file() for name in ("tracked.mp4", "tracks.csv", "tracks.json")):
            self.result_video = result
            self.status.setText("보호색 추적 완료 — 결과 영상·tracks.csv·설정 JSON을 저장했습니다. 정답 기반 정확도 평가는 아닙니다.")
            if self.diagnostics.isChecked():
                summary = self.current_output/"diagnostic_summary.json"
                try:
                    complete = json.loads(summary.read_text(encoding="utf-8")).get("status") == "completed"
                    complete = complete and all((self.current_output/name).is_file() for name in ("diagnostic_report.html", "diagnostic.mp4"))
                except (ValueError, OSError, AttributeError):
                    complete = False
                if complete:
                    self.diagnostic_root = self.current_output
                    self.status.setText("추적·진단 저장 완료 — 진단 영상과 요약에서 의심 구간을 확인하세요. 정확도 평가는 아닙니다.")
                else:
                    self.status.setText("추적 완료 · 진단 기록은 미완료 또는 누락 상태입니다.")
        else:
            self.status.setText(f"추적 실패 (종료 코드 {code}) — 아래 오류 로그를 확인하세요.")
        self.set_busy(False)
        if self.result_video:
            self.play_result()
        self.processing_finished.emit()
        if self.close_pending:
            self.close()

    def stop_processing(self):
        if self.is_busy():
            self.cancelled = True
            self.status.setText("추적 중지 중…")
            process = self.process
            process.terminate()
            QTimer.singleShot(3000, process, lambda: process.kill() if process.state() != QProcess.NotRunning else None)

    def stop_all(self):
        self.player.stop()
        self.stop_processing()

    def position_changed(self, position):
        if not self.slider.isSliderDown():
            self.slider.setValue(position)
        def clock(ms):
            seconds = max(0, ms)//1000
            return f"{seconds//60:02d}:{seconds%60:02d}"
        self.clock.setText(f"{clock(position)} / {clock(self.player.duration())}")

    def apply_learning_inputs(self, dialog):
        if self.dataset_root:
            dialog.data.setText(str(self.dataset_root/"dataset.yaml"))
        if self.model_path:
            dialog.weights.setText(str(self.model_path))
        if self.video_path:
            dialog.video.setText(str(self.video_path))

    def open_learning(self):
        from camouflage_learning_gui import LearningRateDialog
        self.player.pause()
        self.learning_dialog = LearningRateDialog(self)
        self.apply_learning_inputs(self.learning_dialog)
        self.learning_dialog.setAttribute(Qt.WA_DeleteOnClose)
        self.learning_dialog.setModal(True)
        self.learning_dialog.show()

    def open_output(self):
        path = getattr(self, "current_output", None)
        if path and path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self.status.setText("아직 저장된 추적 결과 폴더가 없습니다.")

    def open_diagnostics(self):
        if self.diagnostic_root:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.diagnostic_root/"diagnostic_report.html")))

    def play_diagnostic(self):
        if self.diagnostic_root:
            self.player.setSource(QUrl.fromLocalFile(str(self.diagnostic_root/"diagnostic.mp4")))
            self.player.play()
            self.video_info.setText("진단 영상: 왼쪽 NMS 후 검출·실제 연결 단계 / 오른쪽 추적 ID·궤적")

    def checked(self, action):
        try:
            action()
        except (ValueError, OSError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "파일 확인", str(error))

    def closeEvent(self, event):
        if self.is_busy():
            self.close_pending = True
            self.stop_processing()
            event.ignore()
            return
        self.player.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    page = CamouflagePage()
    page.resize(1400, 900)
    page.show()
    sys.exit(app.exec())
