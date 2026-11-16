"""메인 GUI의 보호색 학습 비교 진입·추적 중 비활성화 검사."""
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
    from PySide6.QtWidgets import QApplication, QPushButton
    import main
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, "PySide6 설치가 필요합니다.")
class LearningEntryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_opens_comparison_and_disables_entry_while_tracking(self):
        # 이 테스트는 메인 오디오/원본 플레이어가 아닌 비교 창 연결만 검사합니다.
        with mock.patch("main.QMediaPlayer"), mock.patch("main.QAudioOutput"):
            window = main.Window()
        try:
            self.assertEqual([window.tabs.tabText(i) for i in range(window.tabs.count())],
                             ["실시간 추적", "성능 비교", "보호색 추적"])
            window.tabs.setCurrentIndex(2)
            self.assertIs(window.learning_button, window.camouflage_page.learning_button)
            self.assertEqual(sum(button.text() == "보호색 학습 비교" for button in window.findChildren(QPushButton)), 1)
            self.assertTrue(window.learning_button.isEnabled())
            window.learning_button.click()
            self.assertTrue(window.learning_dialog.isVisible())
            self.assertTrue(window.learning_dialog.isModal())
            self.assertEqual(window.learning_dialog.weights.text(), str(window.camouflage_page.model_path))
            window.learning_dialog.close()
            window.set_settings_enabled(False)
            self.assertFalse(window.learning_button.isEnabled())
            self.app.processEvents()
        finally:
            window.close()

    def test_closing_main_stops_background_camouflage_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"run_camouflage.py").write_text("import time; time.sleep(30)")
            with mock.patch("main.QMediaPlayer"), mock.patch("main.QAudioOutput"), \
                    mock.patch("camouflage_tracking_page.ROOT", root):
                window = main.Window()
            window.show()
            page = window.camouflage_page
            page.model_path, page.video_path = root/"fake.pt", root/"fake.mp4"
            with mock.patch("camouflage_tracking_page.ROOT", root):
                page.start_tracking()
            try:
                deadline = time.monotonic()+5
                while page.process.state() != QProcess.Running and time.monotonic() < deadline:
                    QTest.qWait(25)
                self.assertTrue(page.is_busy())
                window.close()
                deadline = time.monotonic()+6
                while page.is_busy() and time.monotonic() < deadline:
                    QTest.qWait(25)
                QTest.qWait(100)
                self.assertFalse(page.is_busy())
                self.assertFalse(window.isVisible())
            finally:
                page.process.kill()
                page.process.waitForFinished(1000)
                window.close()
