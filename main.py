"""지능형 객체 추적 시스템 GUI (요구 사항 9).

* 실시간 추적 탭: 영상/카메라 선택, 모델·추적기·대상 클래스 선택, 시작·일시정지·중지,
  박스·ID·궤적 표시, 객체별 상태(가림·화면 이탈·재등장)·방향·거리·속도, FPS·처리 시간.
* 결과 저장/불러오기: 추적 결과 CSV(+설정 JSON) 저장, 저장한 결과를 영상 위에 다시 재생.
* 성능 비교 탭: run_mot.py / benchmark_speed.py가 만든 비교표를 표시.
"""
import json
import sys
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSlider, QSpinBox, QStackedWidget, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from pipeline import (MODEL_DIR, ROOT, TRACKABLE_CLASSES, PipelineConfig, available_weights,
                      default_weights, tracker_choices)
from results_io import ResultWriter, load_results
from tracker import STATUS_EXITED, STATUS_OCCLUDED, STATUS_REAPPEARED, STATUS_REENTERED
from tracking import TrackingThread

MODE_TRACK, MODE_ORIGINAL, MODE_REPLAY = "추적", "원본 재생", "결과 재생"
CLASS_LABELS = {"person": "사람", "cat": "고양이", "dog": "개", "bird": "새"}
TRACKER_LABELS = {"baseline": "ByteTrack 기본", "improved": "ByteTrack 개선"}

STYLE = """
QWidget { background:#101923; color:#edf4fa; font-family:'Malgun Gothic'; font-size:14px; }
QMainWindow { background:#101923; }
QLabel#title { font-size:27px; font-weight:bold; }
QLabel#stat { font-size:13px; }
QPushButton,QComboBox,QSpinBox { background:#223140; border:1px solid #496073; border-radius:7px; padding:8px 14px; }
QPushButton:hover { border-color:#00d4e9; background:#293f50; }
QPushButton:disabled,QComboBox:disabled,QCheckBox:disabled { color:#687786; }
QPushButton#primary { background:#00cddd; color:#081922; font-weight:bold; }
QTabWidget::pane { border:1px solid #344a5c; }
QTabBar::tab { background:#223140; padding:12px 26px; }
QTabBar::tab:selected { background:#00cddd; color:#081922; font-weight:bold; }
QGroupBox { border:1px solid #344a5c; border-radius:8px; margin-top:14px; padding:12px; }
QGroupBox::title { subcontrol-origin:margin; left:14px; }
QTableWidget { background:#16222e; gridline-color:#344a5c; border:0; }
QHeaderView::section { background:#223140; padding:7px; border:0; }
QSlider::groove:horizontal { height:6px; background:#435466; border-radius:3px; }
QSlider::handle:horizontal { background:#00d4e9; width:16px; margin:-5px 0; border-radius:8px; }
QCheckBox { padding:5px; }
"""


def tracker_label(name):
    if name in TRACKER_LABELS:
        return TRACKER_LABELS[name]
    return name if "+" in name else f"튜닝: {name}"


def track_color(track_id):
    return QColor.fromHsv((track_id * 137) % 360, 175, 255)


class TrackingView(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.rows, self.waiting = [], []
        self.display = {}

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0a1118"))
        if self.image is None:
            return
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        painter.translate((self.width() - width) / 2, (self.height() - height) / 2)
        painter.scale(scale, scale)
        rect = QRectF(0, 0, self.image.width(), self.image.height())
        painter.setClipRect(rect)
        painter.drawImage(rect, self.image)
        painter.setRenderHint(QPainter.Antialiasing)
        font = painter.font()
        font.setPixelSize(max(13, round(14 / scale)))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        padding = 4 / scale

        if self.display.get("가림·이탈"):
            for row in self.waiting:
                x1, y1, x2, y2 = row.box
                pen = QPen(QColor("#8a98a6"))
                pen.setWidthF(1.5 / scale)
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                painter.drawText(QPointF(x1, y2 + metrics.ascent()), f"ID {row.track_id} {row.status}")

        for row in self.rows:
            x1, y1, x2, y2 = row.box
            color = track_color(row.track_id)
            pen = QPen(color)
            pen.setWidthF(2 / scale)
            painter.setPen(pen)
            if self.display.get("이동 궤적") and len(row.trail) > 1:
                painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in row.trail]))
            if self.display.get("박스"):
                if row.status in (STATUS_REAPPEARED, STATUS_REENTERED):
                    pen.setWidthF(4 / scale)
                    painter.setPen(pen)
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            if self.display.get("ID"):
                label = f"ID {row.track_id}"
                if row.cls_name != "person":
                    label += f" {CLASS_LABELS.get(row.cls_name, row.cls_name)}"
                if row.status in (STATUS_REAPPEARED, STATUS_REENTERED):
                    label += f" · {row.status}"
                label_height = metrics.height() + 2 * padding
                label_rect = QRectF(x1, max(0, y1 - label_height),
                                    metrics.horizontalAdvance(label) + 2 * padding, label_height)
                label_rect.moveLeft(min(x1, max(0, rect.width() - label_rect.width())))
                painter.fillRect(label_rect, color)
                painter.setPen(QColor("#101923"))
                painter.drawText(label_rect, Qt.AlignCenter, label)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("지능형 객체 추적 시스템")
        self.resize(1440, 900)
        self.source = None          # 파일 경로(str) 또는 카메라 번호(int)
        self.worker = None
        self.stopping = self.closing = self.tracking_failed = False
        self.replay = None
        self.results = ResultWriter()
        self.result_meta = {}
        self.timing_sum = {}
        self.timing_frames = 0

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("지능형 객체 추적 시스템")
        title.setObjectName("title")
        layout.addWidget(title)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(self.build_tracking_tab(), "실시간 추적")
        self.tabs.addTab(self.build_comparison_tab(), "성능 비교")
        self.tabs.currentChanged.connect(lambda index: index == 1 and self.load_comparison())

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setMuted(True)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.errorOccurred.connect(self.media_error)
        self.mode.currentTextChanged.connect(self.mode_changed)
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.statusBar().showMessage("영상 파일을 열거나 카메라를 연결해 주세요.")
        self.enable_controls(False)
        self.update_display()

    # ------------------------------------------------------------------ 화면 구성
    def build_tracking_tab(self):
        page = QWidget()
        main = QVBoxLayout(page)

        source_bar = QHBoxLayout()
        self.open_button = self.button(source_bar, "영상 열기", self.open_video)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 9)
        self.camera_index.setPrefix("카메라 ")
        source_bar.addWidget(self.camera_index)
        self.camera_button = self.button(source_bar, "카메라 연결", self.open_camera)
        source_bar.addWidget(QLabel("모드"))
        self.mode = QComboBox()
        self.mode.addItems([MODE_TRACK, MODE_ORIGINAL, MODE_REPLAY])
        source_bar.addWidget(self.mode)
        source_bar.addStretch()
        self.save_result_button = self.button(source_bar, "결과 저장", self.save_results)
        self.load_result_button = self.button(source_bar, "결과 불러오기", self.load_results_file)
        self.load_button = self.button(source_bar, "설정 불러오기", self.load_settings)
        self.save_button = self.button(source_bar, "설정 저장", self.save_settings)
        main.addLayout(source_bar)

        model_bar = QHBoxLayout()
        model_bar.addWidget(QLabel("검출 모델"))
        self.model = QComboBox()
        self.model.setMinimumWidth(240)
        for path in available_weights():
            self.model.addItem(Path(path).name if Path(path).parent == ROOT else
                               f"{Path(path).parent.name}/{Path(path).name}", path)
        if self.model.findData(default_weights()) < 0:
            self.model.addItem(Path(default_weights()).name, default_weights())
        self.model.setCurrentIndex(max(0, self.model.findData(default_weights())))
        model_bar.addWidget(self.model)
        self.model_file_button = self.button(model_bar, "…", self.choose_model_file)
        model_bar.addWidget(QLabel("추적기"))
        self.tracker = QComboBox()
        for name in tracker_choices():
            self.tracker.addItem(tracker_label(name), name)
        self.tracker.setCurrentIndex(max(0, self.tracker.findData("improved")))
        model_bar.addWidget(self.tracker)
        model_bar.addWidget(QLabel("대상"))
        self.class_checks = {}
        for name in TRACKABLE_CLASSES:
            check = QCheckBox(CLASS_LABELS[name])
            check.setChecked(name == "person")
            model_bar.addWidget(check)
            self.class_checks[name] = check
        self.mosaic_check = QCheckBox("얼굴 모자이크")
        self.mosaic_check.setChecked((MODEL_DIR / "face.pt").is_file())
        self.mosaic_check.setEnabled((MODEL_DIR / "face.pt").is_file())
        model_bar.addWidget(self.mosaic_check)
        model_bar.addStretch()
        main.addLayout(model_bar)
        self.setting_widgets = [self.model, self.model_file_button, self.tracker, self.mosaic_check,
                                *self.class_checks.values()]

        body = QHBoxLayout()
        left = QVBoxLayout()
        self.video = QVideoWidget()
        self.preview = TrackingView()
        self.video_stack = QStackedWidget()
        self.video_stack.setMinimumSize(560, 340)
        self.video_stack.addWidget(self.video)
        self.video_stack.addWidget(self.preview)
        self.video_stack.setCurrentWidget(self.preview)
        left.addWidget(self.video_stack, 1)
        controls = QHBoxLayout()
        self.play_button = self.button(controls, "▶ 시작", self.play)
        self.play_button.setObjectName("primary")
        self.pause_button = self.button(controls, "Ⅱ 일시정지", self.pause)
        self.stop_button = self.button(controls, "■ 중지", self.stop)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        controls.addWidget(self.slider, 1)
        self.clock = QLabel("00:00 / 00:00")
        controls.addWidget(self.clock)
        left.addLayout(controls)
        body.addLayout(left, 3)

        panel = QVBoxLayout()
        stats = QGroupBox("추적 상태")
        stats_layout = QHBoxLayout(stats)
        self.stats = [QLabel() for _ in range(5)]
        for stat in self.stats:
            stat.setObjectName("stat")
            stats_layout.addWidget(stat)
        self.set_stats()
        panel.addWidget(stats)
        objects = QGroupBox("객체 목록")
        objects_layout = QVBoxLayout(objects)
        self.objects = self.table(["ID", "클래스", "상태", "방향", "거리(px)", "속도(px/s)"])
        objects_layout.addWidget(self.objects)
        panel.addWidget(objects, 1)
        self.options = {}
        options = QHBoxLayout()
        for text in ["박스", "ID", "이동 궤적", "가림·이탈"]:
            check = QCheckBox(text)
            check.setChecked(True)
            check.toggled.connect(self.update_display)
            options.addWidget(check)
            self.options[text] = check
        panel.addLayout(options)
        body.addLayout(panel, 2)
        main.addLayout(body, 1)
        return page

    def build_comparison_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        bar = QHBoxLayout()
        self.comparison_label = QLabel()
        bar.addWidget(self.comparison_label, 1)
        self.button(bar, "새로고침", self.load_comparison)
        self.button(bar, "비교 파일 열기", self.open_comparison)
        layout.addLayout(bar)
        layout.addWidget(QLabel("추적 성능 (MOT 정답 기준, 굵은 글씨 = 열별 최고)"))
        self.comparison = self.table(["실험", "검출 모델", "추적기", "HOTA", "IDF1", "MOTA", "IDSW",
                                      "Frag", "FPS", "평균 ms"])
        layout.addWidget(self.comparison, 2)
        layout.addWidget(QLabel("처리 속도 요인 (benchmark_speed.py)"))
        self.speed = self.table(["조건", "검출 ms", "추적 ms", "모자이크 ms", "전체 ms", "FPS", "평균 객체 수"])
        layout.addWidget(self.speed, 1)
        self.comparison_path = ROOT / "results" / "comparison.json"
        self.load_comparison()
        return page

    @staticmethod
    def button(layout, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    @staticmethod
    def table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().hide()
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        return table

    @staticmethod
    def fill(table, rows, bold=None):
        table.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                if bold and (r, c) in bold:
                    font = QFont(item.font())
                    font.setBold(True)
                    item.setFont(font)
                    item.setForeground(QBrush(QColor("#00d4e9")))
                table.setItem(r, c, item)

    # ------------------------------------------------------------------ 입력 선택
    def enable_controls(self, enabled):
        for widget in (self.play_button, self.pause_button, self.stop_button, self.slider):
            widget.setEnabled(enabled)
        self.save_result_button.setEnabled(bool(len(self.results)) and self.worker is None)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "영상 열기", "", "영상 (*.mp4 *.avi *.mkv *.mov *.webm);;모든 파일 (*)")
        if path:
            self.set_source(str(Path(path).resolve()))

    def open_camera(self):
        if self.mode.currentText() != MODE_TRACK:
            self.mode.setCurrentText(MODE_TRACK)
        self.set_source(self.camera_index.value())

    def set_source(self, source):
        self.source = source
        self.replay = None if self.mode.currentText() != MODE_REPLAY else self.replay
        self.mode_changed()
        name = f"카메라 {source}" if isinstance(source, int) else Path(source).name
        self.statusBar().showMessage(f"{name} · 시작을 눌러 주세요.")

    def choose_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "검출 가중치 선택", str(MODEL_DIR), "YOLO 가중치 (*.pt)")
        if path:
            if self.model.findData(path) < 0:
                self.model.addItem(Path(path).name, path)
            self.model.setCurrentIndex(self.model.findData(path))

    def pipeline_config(self):
        classes = [name for name, check in self.class_checks.items() if check.isChecked()]
        if not classes:
            raise ValueError("추적할 대상을 하나 이상 선택하세요.")
        return PipelineConfig(weights=self.model.currentData(), classes=classes,
                              tracker=self.tracker.currentData(), mosaic=self.mosaic_check.isChecked())

    def mode_changed(self):
        self.player.stop()
        self.clear_tracking()
        mode = self.mode.currentText()
        camera = isinstance(self.source, int)
        if mode != MODE_TRACK and camera:
            self.source = None
        if mode == MODE_REPLAY and self.replay is None:
            self.statusBar().showMessage("결과 불러오기로 저장한 추적 결과 CSV를 선택하세요.")
        self.enable_controls(self.source is not None and (mode != MODE_REPLAY or self.replay is not None))
        original = mode == MODE_ORIGINAL
        self.video_stack.setCurrentWidget(self.video if original else self.preview)
        self.slider.setEnabled(original and self.source is not None)
        for widget in self.setting_widgets:
            widget.setEnabled(mode == MODE_TRACK)
        self.player.setSource(QUrl.fromLocalFile(self.source) if original and self.source else QUrl())

    # ------------------------------------------------------------------ 재생 제어
    def play(self):
        if self.source is None or self.stopping:
            return
        mode = self.mode.currentText()
        if mode == MODE_ORIGINAL:
            if self.player.duration() and self.player.position() >= self.player.duration():
                self.player.setPosition(0)
            self.player.play()
            self.statusBar().showMessage("원본 영상 재생 중")
            return
        if self.worker:
            self.worker.set_paused(False)
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.statusBar().showMessage("추적 중" if mode == MODE_TRACK else "결과 재생 중")
            return
        try:
            config = self.pipeline_config() if mode == MODE_TRACK else None
        except ValueError as error:
            QMessageBox.warning(self, "설정 오류", str(error))
            return

        self.clear_tracking()
        self.tracking_failed = False
        if mode == MODE_TRACK:
            self.results = ResultWriter()
            self.result_meta = dict(source=str(self.source), pipeline=config.to_dict())
        self.worker = TrackingThread(self.source, config, self.replay if mode == MODE_REPLAY else None, self)
        self.worker.metadata.connect(self.tracking_metadata)
        self.worker.frame_ready.connect(self.show_tracks)
        self.worker.failed.connect(self.tracking_error)
        self.worker.finished.connect(self.tracking_finished)
        for widget in (self.open_button, self.camera_button, self.mode, self.load_button,
                       self.load_result_button, self.save_result_button, self.play_button,
                       *self.setting_widgets):
            widget.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("모델 준비 중 · 첫 실행 시 가중치를 다운로드할 수 있습니다."
                                     if mode == MODE_TRACK else "결과 재생 준비 중")
        self.worker.start()

    def pause(self):
        if self.worker and not self.stopping:
            self.worker.set_paused(True)
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
        else:
            self.player.pause()
        self.statusBar().showMessage("일시정지")

    def stop(self):
        if self.worker:
            self.stopping = True
            self.enable_controls(False)
            self.worker.cancel()
            self.statusBar().showMessage("중지 중 · 현재 프레임 처리가 끝날 때까지 기다려 주세요.")
        elif self.mode.currentText() == MODE_ORIGINAL:
            self.player.stop()
            self.statusBar().showMessage("중지")
        else:
            self.clear_tracking()
            self.statusBar().showMessage("중지 · 시작하면 처음부터 추적합니다.")

    def clear_tracking(self):
        self.preview.image = None
        self.preview.rows, self.preview.waiting = [], []
        self.preview.update()
        self.objects.setRowCount(0)
        self.slider.setRange(0, 0)
        self.clock.setText("00:00 / 00:00")
        self.timing_sum, self.timing_frames = {}, 0
        self.set_stats()

    # ------------------------------------------------------------------ 추적 결과 표시
    def set_stats(self, count=None, waiting=None, timings=None):
        values = ["—"] * 5
        if timings is not None:
            values[0] = str(count)
            values[1] = str(waiting)
            if timings:
                n = max(1, self.timing_frames)
                average = self.timing_sum["total"] / n
                values[2] = f"{1000 / max(timings['total'], 1e-6):.1f}"
                values[3] = f"{1000 / max(average, 1e-6):.1f}\n{average:.1f} ms"
                values[4] = (f"검출 {self.timing_sum['detect'] / n:.0f} · 추적 {self.timing_sum['track'] / n:.1f}"
                             f"\n모자이크 {self.timing_sum['mosaic'] / n:.0f} ms")
        labels = ("추적 중", "가림·이탈", "현재 FPS", "평균 FPS", "평균 단계별 시간")
        for widget, label, value in zip(self.stats, labels, values):
            widget.setText(f"{label}\n\n{value}")

    def update_display(self):
        self.preview.display = {key: check.isChecked() for key, check in self.options.items()}
        self.preview.update()

    def tracking_metadata(self, duration, fps, camera):
        self.result_meta.update(fps=fps, camera=camera)
        if not self.stopping:
            self.slider.setRange(0, duration)
            self.clock.setText("실시간" if camera else f"00:00 / {self.time_text(duration)}")

    def show_tracks(self, image, rows, waiting, position, frame_index, timings):
        worker = self.worker
        if worker is None:
            return
        try:
            if self.stopping:
                return
            if timings:
                self.timing_frames += 1
                for key, value in timings.items():
                    self.timing_sum[key] = self.timing_sum.get(key, 0.0) + value
            if self.mode.currentText() == MODE_TRACK:
                self.results.add(frame_index, position, rows + waiting)
            if frame_index == 1 or (frame_index % 30 == 0 and self.pause_button.isEnabled()):
                self.statusBar().showMessage(
                    "추적 중" if self.mode.currentText() == MODE_TRACK else "결과 재생 중")
            self.preview.image = image
            self.preview.rows, self.preview.waiting = rows, waiting
            self.preview.update()
            if self.slider.maximum():
                self.slider.setValue(position)
                self.clock.setText(f"{self.time_text(position)} / {self.time_text(self.slider.maximum())}")
            else:
                self.clock.setText(f"실시간 · {self.time_text(position)}")
            self.set_stats(len(rows), len(waiting), timings)
            table_rows = [(str(row.track_id), CLASS_LABELS.get(row.cls_name, row.cls_name),
                           row.status, row.direction, f"{row.distance:.1f}",
                           f"{row.speed:.0f}" if index < len(rows) else "—")
                          for index, row in enumerate(rows + waiting)]
            self.fill(self.objects, table_rows)
            for r in range(len(rows), len(table_rows)):  # 가림·이탈 트랙은 흐리게
                for c in range(self.objects.columnCount()):
                    self.objects.item(r, c).setForeground(QBrush(QColor("#8a98a6")))
        finally:
            worker.displayed.set()

    def tracking_error(self, message):
        self.tracking_failed = True
        if not self.stopping:
            self.statusBar().showMessage("추적 실패")
            # 비모달 대화상자로 완료 신호가 처리되는 동안 UI를 막지 않습니다.
            warning = QMessageBox(QMessageBox.Warning, "추적 실패", message, parent=self)
            warning.setAttribute(Qt.WA_DeleteOnClose)
            warning.show()

    def tracking_finished(self):
        if self.worker and self.worker.tracker_config is not None:
            self.result_meta["tracker_config"] = self.worker.tracker_config.to_dict()
        self.worker.deleteLater()
        self.worker = None
        if self.closing:
            self.close()
            return
        for widget in (self.open_button, self.camera_button, self.mode, self.load_button,
                       self.load_result_button, self.play_button):
            widget.setEnabled(True)
        for widget in self.setting_widgets:
            widget.setEnabled(self.mode.currentText() == MODE_TRACK)
        self.mosaic_check.setEnabled(self.mosaic_check.isEnabled() and (MODEL_DIR / "face.pt").is_file())
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_result_button.setEnabled(bool(len(self.results)))
        saved = f" · 결과 {len(self.results)}행, '결과 저장'으로 파일에 저장할 수 있습니다." if len(self.results) else ""
        if self.stopping:
            self.statusBar().showMessage("중지" + saved)
        elif not self.tracking_failed:
            duration = self.slider.maximum()
            if duration:
                self.slider.setValue(duration)
                self.clock.setText(f"{self.time_text(duration)} / {self.time_text(duration)}")
            self.statusBar().showMessage("완료" + saved)
        self.stopping = False

    @staticmethod
    def time_text(milliseconds):
        seconds = milliseconds // 1000
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    # ------------------------------------------------------------------ 원본 재생
    def position_changed(self, position):
        if self.mode.currentText() != MODE_ORIGINAL:
            return
        if not self.slider.isSliderDown():
            self.slider.setValue(position)
        self.clock.setText(f"{self.time_text(position)} / {self.time_text(self.player.duration())}")

    def duration_changed(self, duration):
        if self.mode.currentText() != MODE_ORIGINAL:
            return
        self.slider.setRange(0, duration)
        self.position_changed(self.player.position())

    def media_error(self, error, message):
        if self.mode.currentText() != MODE_ORIGINAL:
            return
        self.enable_controls(False)
        self.statusBar().showMessage("영상 오류 · 다른 파일을 열어 주세요.")
        QMessageBox.warning(self, "영상 재생 오류", message or "지원하지 않는 영상입니다.")

    # ------------------------------------------------------------------ 결과 파일 (요구 사항 10)
    def save_results(self):
        if not len(self.results):
            return
        default = "tracking_results.csv"
        if isinstance(self.source, str):
            default = f"{Path(self.source).stem}_tracks.csv"
        path, _ = QFileDialog.getSaveFileName(self, "추적 결과 저장", default, "CSV (*.csv)")
        if not path:
            return
        try:
            self.results.save(path, dict(self.result_meta, rows=len(self.results)))
            self.statusBar().showMessage(f"저장: {path} (+ {Path(path).with_suffix('.json').name})")
        except OSError as error:
            QMessageBox.warning(self, "저장 실패", str(error))

    def load_results_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "추적 결과 불러오기", "", "CSV (*.csv)")
        if not path:
            return
        try:
            frames, metadata = load_results(path)
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "불러오기 실패", str(error))
            return
        video = metadata.get("source")
        if not video or not Path(video).is_file():
            video, _ = QFileDialog.getOpenFileName(
                self, "결과에 해당하는 영상 선택", "", "영상 (*.mp4 *.avi *.mkv *.mov *.webm)")
            if not video:
                return
        self.replay = frames
        self.mode.blockSignals(True)
        self.mode.setCurrentText(MODE_REPLAY)
        self.mode.blockSignals(False)
        self.source = str(Path(video).resolve())
        self.mode_changed()
        objects = len({row["id"] for rows in frames.values() for row in rows})
        tracker = metadata.get("pipeline", {}).get("tracker", "?")
        self.statusBar().showMessage(f"결과 불러옴: {len(frames)}프레임, 객체 {objects}개, 추적기 {tracker} · 시작을 누르세요.")

    # ------------------------------------------------------------------ 설정 파일 (요구 사항 10)
    def save_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, "설정 저장", "settings.json", "JSON (*.json)")
        if not path:
            return
        try:
            config = self.pipeline_config()
            data = {"version": 2, "mode": self.mode.currentText(), "pipeline": config.to_dict(),
                    "camera": self.camera_index.value(),
                    "display": {key: value.isChecked() for key, value in self.options.items()}}
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "저장 실패", str(error))

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "설정 불러오기", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") not in (1, 2):
                raise ValueError("지원하지 않는 설정 형식입니다.")
            display = data.get("display")
            if not isinstance(display, dict) or any(
                    type(display.get(key, True)) is not bool for key in self.options):
                raise ValueError("표시 옵션은 true 또는 false여야 합니다.")
            if data["version"] == 2:
                config = PipelineConfig.from_dict(data.get("pipeline", {}))
                if self.model.findData(config.weights) < 0:
                    if not Path(config.weights).is_file():
                        raise ValueError(f"가중치 파일이 없습니다: {config.weights}")
                    self.model.addItem(Path(config.weights).name, config.weights)
                self.model.setCurrentIndex(self.model.findData(config.weights))
                if self.tracker.findData(config.tracker) < 0:
                    raise ValueError(f"추적기 설정이 없습니다: {config.tracker}")
                self.tracker.setCurrentIndex(self.tracker.findData(config.tracker))
                for name, check in self.class_checks.items():
                    check.setChecked(name in config.classes)
                self.mosaic_check.setChecked(config.mosaic and self.mosaic_check.isEnabled())
                self.camera_index.setValue(int(data.get("camera", 0)))
                mode = data.get("mode", MODE_TRACK)
                self.mode.setCurrentText(mode if mode != MODE_REPLAY else MODE_TRACK)
            else:  # 이전 버전 설정: 추적기 이름과 표시 옵션만 있음
                self.mode.setCurrentText(MODE_ORIGINAL if data.get("tracker") == "원본 재생" else MODE_TRACK)
            for key, check in self.options.items():
                check.setChecked(display.get(key, True))
        except (OSError, ValueError, TypeError, KeyError) as error:
            QMessageBox.warning(self, "불러오기 실패", str(error))

    # ------------------------------------------------------------------ 성능 비교 탭
    def open_comparison(self):
        path, _ = QFileDialog.getOpenFileName(self, "비교 결과 열기", str(ROOT / "results"), "JSON (*.json)")
        if path:
            self.comparison_path = Path(path)
            self.load_comparison()

    def load_comparison(self):
        entries = []
        try:
            if self.comparison_path.is_file():
                entries = json.loads(self.comparison_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            self.comparison_label.setText(f"비교 파일을 읽지 못했습니다: {error}")
            return
        rows, bold = [], set()
        columns = {3: "HOTA", 4: "IDF1", 5: "MOTA", 8: "fps"}
        for c, key in columns.items():
            if entries:
                best = max(e[key] for e in entries)
                bold |= {(r, c) for r, e in enumerate(entries) if e[key] == best}
        if entries:
            best = min(e["IDSW"] for e in entries)
            bold |= {(r, 6) for r, e in enumerate(entries) if e["IDSW"] == best}
        for e in entries:
            rows.append((e["name"], e["weights"], tracker_label(e["tracker"]),
                         f"{e['HOTA']:.2f}", f"{e['IDF1']:.2f}", f"{e['MOTA']:.2f}", str(e["IDSW"]),
                         str(e["Frag"]), f"{e['fps']:.1f}", f"{e['total_ms']:.1f}"))
        self.fill(self.comparison, rows, bold)
        self.comparison_label.setText(
            f"{self.comparison_path} · {len(entries)}개 결과" if entries else
            "비교 결과가 없습니다. run_mot.py로 MOT 평가를 실행하면 여기에 표시됩니다.")

        speed_path = ROOT / "results" / "speed.json"
        speed_rows = []
        if speed_path.is_file():
            try:
                for r in json.loads(speed_path.read_text(encoding="utf-8"))["results"]:
                    speed_rows.append((r["label"], f"{r['detect']:.1f}", f"{r['track']:.2f}",
                                       f"{r['mosaic']:.1f}", f"{r['total']:.1f}", f"{r['fps']:.1f}",
                                       f"{r['mean_objects']:.1f}"))
            except (OSError, ValueError, KeyError):
                pass
        self.fill(self.speed, speed_rows)

    def closeEvent(self, event):
        if self.worker:
            self.closing = True
            self.stop()
            event.ignore()
            return
        self.player.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Window()
    window.show()
    sys.exit(app.exec())
