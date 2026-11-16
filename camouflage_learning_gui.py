"""보호색 학습률 비교 창. 단독 실행하거나 메인 GUI에서 열 수 있습니다."""
import codecs
import json
import math
import re
import sys
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

ROOT = Path(__file__).resolve().parent


class LearningRateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("보호색 동물 — 학습률 재학습·영상 비교")
        self.resize(1120, 900)
        self.process = None
        self.report_root = None
        self.cancelled = self.close_when_finished = False
        layout = QVBoxLayout(self)
        notice = QLabel("학습률 변경은 재학습이 필요합니다. 정답 라벨이 있는 train/val 데이터가 필요하며 CPU에서는 오래 걸릴 수 있습니다.\n"
                        "영상은 학습 정답이 아닌 참고 비교용입니다. 후보 선택은 val mAP50-95로 하며 test는 평가하지 않습니다.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.form_widget = QWidget()
        form = QFormLayout(self.form_widget)
        self.data = self.path_field(form, "데이터 dataset.yaml", ROOT/"datasets/camouflage/dataset.yaml", "YAML (*.yaml *.yml)")
        self.weights = self.path_field(form, "모든 후보의 초기 가중치", ROOT/"yolo11n.pt", "모델 (*.pt)")
        self.video = self.path_field(form, "비교할 원본 영상", "", "영상 (*.mp4 *.avi *.mov *.mkv)")
        self.output = self.path_field(form, "결과 저장 상위 폴더", ROOT/"training_runs", directory=True)
        self.rates = QLineEdit("0.0001, 0.0005")
        form.addRow("비교할 초기 학습률 (2~6개)", self.rates)
        settings = QHBoxLayout()
        self.epochs = self.spin(settings, "에폭", 1, 1000, 3)
        self.batch = self.spin(settings, "배치", 1, 128, 8)
        self.imgsz = self.spin(settings, "입력 크기", 32, 2048, 640)
        self.device = QComboBox()
        self.device.addItem("CPU", "cpu")
        self.device.addItem("GPU 0 (CUDA 필요)", "0")
        settings.addWidget(self.device)
        form.addRow(settings)
        layout.addWidget(self.form_widget)
        actions = QHBoxLayout()
        self.start_button = QPushButton("재학습 비교 시작")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("학습 중지")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        self.load_button = QPushButton("기존 비교 결과 불러오기")
        self.load_button.clicked.connect(self.choose_report)
        folder = QPushButton("결과 폴더 열기")
        folder.clicked.connect(self.open_result_folder)
        for button in (self.start_button, self.stop_button, self.load_button, folder):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.status = QLabel("대기 — 기존 3에폭 모델은 덮어쓰지 않습니다.")
        layout.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMaximumHeight(130)
        layout.addWidget(self.log)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["초기 학습률", "val mAP50 (%)", "val mAP50-95 (%)", "출력 ID 수", "겹침 프레임"])
        self.table.setMaximumHeight(150)
        layout.addWidget(self.table)
        self.selectors, self.players, self.screens = [], [], []
        panels = QHBoxLayout()
        for side in ("왼쪽 후보", "오른쪽 후보"):
            panel = QVBoxLayout()
            panel.addWidget(QLabel(side))
            selector = QComboBox()
            player = QMediaPlayer(self)
            screen = QVideoWidget()
            screen.setMinimumSize(360, 200)
            player.setVideoOutput(screen)
            selector.currentIndexChanged.connect(lambda _, s=selector, p=player: p.setSource(
                QUrl.fromLocalFile(s.currentData()) if s.currentData() and Path(s.currentData()).is_file() else QUrl()))
            player.errorOccurred.connect(lambda _, message: self.status.setText(f"영상 재생 오류: {message}"))
            panel.addWidget(selector)
            panel.addWidget(screen)
            panels.addLayout(panel)
            self.selectors.append(selector)
            self.players.append(player)
            self.screens.append(screen)
        layout.addLayout(panels)
        playback = QHBoxLayout()
        play = QPushButton("선택한 두 영상 처음부터 재생")
        play.clicked.connect(self.play_videos)
        pause = QPushButton("두 영상 일시정지")
        pause.clicked.connect(lambda: [player.pause() for player in self.players])
        playback.addWidget(play)
        playback.addWidget(pause)
        layout.addLayout(playback)
        layout.addWidget(QLabel("두 재생기는 프레임 단위 동기화를 보장하지 않습니다. 겹침·ID 수는 정확도 지표가 아닙니다."))

    def path_field(self, form, title, value, pattern="", directory=False):
        row = QHBoxLayout()
        field = QLineEdit(str(value))
        button = QPushButton("선택")
        def choose():
            path = QFileDialog.getExistingDirectory(self, title) if directory else QFileDialog.getOpenFileName(self, title, str(ROOT), pattern)[0]
            if path:
                field.setText(path)
        button.clicked.connect(choose)
        row.addWidget(field)
        row.addWidget(button)
        form.addRow(title, row)
        return field

    @staticmethod
    def spin(layout, label, minimum, maximum, value):
        layout.addWidget(QLabel(label))
        field = QSpinBox()
        field.setRange(minimum, maximum)
        field.setValue(value)
        layout.addWidget(field)
        return field

    def command(self):
        rates = [float(value) for value in self.rates.text().replace(",", " ").split()]
        if not 2 <= len(rates) <= 6 or len(set(rates)) != len(rates) or any(not math.isfinite(v) or not 0 < v <= 1 for v in rates):
            raise ValueError("서로 다른 유한한 학습률 2~6개를 0 초과 1 이하로 입력하세요.")
        for field in (self.data, self.weights, self.video):
            if not Path(field.text()).is_file():
                raise ValueError(f"파일을 찾을 수 없습니다: {field.text()}")
        if not self.output.text().strip():
            raise ValueError("결과 저장 상위 폴더를 선택하세요.")
        output = Path(self.output.text()).resolve() / f"lr_compare_{uuid4().hex[:8]}"
        arguments = ["-u", str(ROOT/"compare_camouflage_learning_rates.py"),
                     "--data", self.data.text(), "--weights", self.weights.text(), "--video", self.video.text(),
                     "--epochs", str(self.epochs.value()), "--batch", str(self.batch.value()),
                     "--imgsz", str(self.imgsz.value()), "--device", self.device.currentData(),
                     "--output", str(output), "--lr", *map(str, rates)]
        return output, arguments

    def busy(self, active):
        self.form_widget.setEnabled(not active)
        self.start_button.setEnabled(not active)
        self.load_button.setEnabled(not active)
        self.stop_button.setEnabled(active)

    def start_training(self):
        if self.process and self.process.state() != QProcess.NotRunning:
            return
        try:
            output, arguments = self.command()
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "입력 확인", str(error))
            return
        for player in self.players:
            player.stop()
        for selector in self.selectors:
            selector.clear()
        self.table.setRowCount(0)
        self.report_root = None
        self.current_output = output
        self.cancelled = False
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.log.clear()
        self.busy(True)
        self.status.setText("데이터 검사 및 학습 실행 중 — 로그를 확인하세요.")
        process = QProcess(self)
        self.process = process
        process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUTF8", "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self.read_log)
        process.finished.connect(self.training_finished)
        process.errorOccurred.connect(self.process_error)
        process.start(sys.executable, arguments)

    def read_log(self):
        text = self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        self.log.appendPlainText(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "\n"))

    def process_error(self, error):
        if error == QProcess.FailedToStart:
            self.busy(False)
            self.status.setText(f"실행 시작 실패: {self.process.errorString()}")

    def stop_training(self):
        process = self.process
        if process and process.state() != QProcess.NotRunning:
            self.cancelled = True
            self.status.setText("학습 중지 중 — 완료된 후보·체크포인트는 결과 폴더에 남습니다.")
            process.terminate()
            QTimer.singleShot(3000, process, lambda: process.kill() if process.state() != QProcess.NotRunning else None)

    def training_finished(self, code, exit_status):
        self.read_log()
        self.busy(False)
        report = self.current_output / "comparison.json"
        if self.cancelled:
            if report.is_file():
                payload = json.loads(report.read_text(encoding="utf-8"))
                if payload.get("status") != "completed":
                    payload["status"] = "cancelled"
                    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.status.setText("사용자가 중지했습니다. 부분 결과를 전체 학습 완료로 해석하지 마세요.")
        elif code == 0 and report.is_file():
            self.load_report(report)
        else:
            self.status.setText(f"실행 실패 (종료 코드 {code}). 위 로그와 결과 폴더의 부분 결과를 확인하세요.")
        if self.close_when_finished:
            self.close()

    def choose_report(self):
        path, _ = QFileDialog.getOpenFileName(self, "학습률 비교 결과", str(ROOT), "비교 결과 (comparison.json)")
        if path:
            try:
                self.load_report(Path(path))
            except (ValueError, KeyError, TypeError, OSError) as error:
                QMessageBox.warning(self, "결과 읽기 실패", str(error))

    def load_report(self, path):
        path = Path(path).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("kind") != "camouflage_learning_rates":
            raise ValueError("학습률 비교 보고서가 아닙니다.")
        prepared = []
        for row in report["candidates"]:
            values = [row["lr0"], row["val"]["mAP50"], row["val"]["mAP50-95"]]
            if not all(type(v) in (float, int) and math.isfinite(v) for v in values):
                raise ValueError("비교 지표가 유한한 수가 아닙니다.")
            if not 0 < values[0] <= 1 or any(not 0 <= value <= 1 for value in values[1:]):
                raise ValueError("학습률 또는 val 지표 범위가 올바르지 않습니다.")
            video = (path.parent / row["video"]).resolve()
            if Path(row["video"]).is_absolute() or not video.is_relative_to(path.parent):
                raise ValueError("비교 폴더 밖 영상 경로입니다.")
            prepared.append((row, video))
        for player in self.players:
            player.stop()
        self.report_root = path.parent
        for selector in self.selectors:
            selector.clear()
        self.table.setRowCount(len(prepared))
        missing = 0
        for index, (row, video) in enumerate(prepared):
            values = [str(row["lr0"]), f"{row['val']['mAP50']*100:.2f}", f"{row['val']['mAP50-95']*100:.2f}",
                      str(row["diagnostics"]["unique_ids"]), str(row["diagnostics"]["overlap_frames"])]
            if report["status"] == "completed" and row.get("index", index) == report.get("best_candidate"):
                values[0] += " (val 선택)"
            for column, value in enumerate(values):
                self.table.setItem(index, column, QTableWidgetItem(value))
            missing += int(not video.is_file())
            for selector in self.selectors:
                selector.addItem(f"학습률 {row['lr0']}", str(video))
        if len(prepared) > 1:
            self.selectors[1].setCurrentIndex(1)
        self.table.resizeColumnsToContents()
        self.status.setText(f"보고서 상태: {report['status']} · 후보 {len(prepared)}개 · 영상 누락 {missing}개. 정답 ID 기반 추적 정확도는 미평가.")

    def play_videos(self):
        if any(not selector.currentData() or not Path(selector.currentData()).is_file() for selector in self.selectors):
            QMessageBox.information(self, "영상 확인", "보고서와 후보별 video 폴더를 함께 준비하세요.")
            return
        for player in self.players:
            player.setPosition(0)
            player.play()

    def open_result_folder(self):
        folder = self.report_root or getattr(self, "current_output", None)
        if folder and folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def closeEvent(self, event):
        if self.process and self.process.state() != QProcess.NotRunning:
            self.close_when_finished = True
            self.stop_training()
            event.ignore()
            return
        for player in self.players:
            player.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = LearningRateDialog()
    dialog.show()
    sys.exit(app.exec())
