"""오프스크린 GUI 입력·결과 표시·비동기 실행/중지 검사. 실제 모델 학습은 하지 않습니다."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QProcess
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from camouflage_learning_gui import LearningRateDialog
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, "PySide6 설치가 필요합니다.")
class LearningGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dialog = LearningRateDialog()
        self.addCleanup(self.dialog.close)
        for field, filename in ((self.dialog.data, "dataset.yaml"), (self.dialog.weights, "initial.pt"), (self.dialog.video, "video.mp4")):
            path = self.root/filename
            path.write_bytes(b"test fixture")
            field.setText(str(path))
        self.dialog.output.setText(str(self.root))

    def test_command_uses_selected_lr_and_same_initial_weights(self):
        self.dialog.rates.setText("0.0002, 0.0008")
        output, args = self.dialog.command()
        self.assertEqual(args[args.index("--lr")+1:], ["0.0002", "0.0008"])
        self.assertEqual(args[args.index("--weights")+1], str(self.root/"initial.pt"))
        self.assertEqual(args[args.index("--device")+1], "cpu")
        self.assertEqual(output.parent, self.root)
        self.dialog.rates.setText("nan, 0.1")
        with self.assertRaises(ValueError):
            self.dialog.command()

    def test_load_portable_report_and_reject_outside_video(self):
        rows = [dict(lr0=rate, val={"mAP50": 0.6, "mAP50-95": 0.4},
                     diagnostics=dict(unique_ids=2, overlap_frames=3), video=f"candidate_{index:02d}/video/tracked.mp4")
                for index, rate in enumerate((0.0001, 0.0005))]
        report = dict(kind="camouflage_learning_rates", status="completed", candidates=rows, best_candidate=1)
        path = self.root/"comparison.json"
        path.write_text(json.dumps(report))
        self.dialog.load_report(path)
        self.assertEqual(self.dialog.table.rowCount(), 2)
        self.assertEqual(self.dialog.table.item(0, 1).text(), "60.00")
        self.assertEqual(self.dialog.selectors[1].currentIndex(), 1)
        self.assertIn("val 선택", self.dialog.table.item(1, 0).text())
        self.assertIn("영상 누락 2개", self.dialog.status.text())
        rows[0]["video"] = "../outside.mp4"
        path.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            self.dialog.load_report(path)

    def test_worker_is_async_and_can_be_cancelled_without_training(self):
        arguments = ["-u", "-c", "import time; print('worker started', flush=True); time.sleep(30)"]
        with mock.patch.object(self.dialog, "command", return_value=(self.root/"result", arguments)):
            self.dialog.start_training()
        try:
            deadline = time.monotonic()+5
            while self.dialog.process.state() != QProcess.Running and time.monotonic() < deadline:
                QTest.qWait(25)
            self.assertEqual(self.dialog.process.state(), QProcess.Running)
            self.assertFalse(self.dialog.start_button.isEnabled())
            self.assertTrue(self.dialog.stop_button.isEnabled())
            self.dialog.stop_training()
            deadline = time.monotonic()+5
            while self.dialog.process.state() != QProcess.NotRunning and time.monotonic() < deadline:
                QTest.qWait(25)
            QTest.qWait(50)
            self.assertEqual(self.dialog.process.state(), QProcess.NotRunning)
            self.assertTrue(self.dialog.cancelled)
            self.assertTrue(self.dialog.start_button.isEnabled())
        finally:
            self.dialog.process.kill()
            self.dialog.process.waitForFinished(1000)
