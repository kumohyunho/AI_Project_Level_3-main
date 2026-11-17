"""실시간 추적 탭과 보호색 추적 탭이 서로의 처리 상태를 알맞게 반영하는지 검사합니다.

* 모드(원본 재생·결과 재생)를 고르는 것만으로는 보호색 탭을 막지 않습니다.
* 한쪽이 영상을 처리하는 동안에는 다른 쪽의 처리를 시작하지 않습니다 (검출기 두 개가 CPU를 나눠 써서 둘 다 느려짐).
"""
import os
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication
    import main
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(QT_AVAILABLE, "PySide6 설치가 필요합니다.")
class GuiBusyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self):
        with mock.patch("main.QMediaPlayer"), mock.patch("main.QAudioOutput"):
            window = main.Window()
        self.addCleanup(window.close)
        return window

    def test_mode_change_does_not_block_camouflage_tab(self):
        window = self.make_window()
        for mode in (main.MODE_ORIGINAL, main.MODE_REPLAY, main.MODE_TRACK):
            window.mode.setCurrentText(mode)
            self.app.processEvents()
            self.assertTrue(window.learning_button.isEnabled(), mode)

    def test_main_tracking_blocks_camouflage_processing(self):
        window = self.make_window()
        page = window.camouflage_page
        page.model_path, page.video_path = ROOT/"models"/"camouflage.pt", ROOT/"sample.mp4"
        page.set_busy(False)
        self.assertTrue(page.track_button.isEnabled())
        window.worker = object()
        window.notify_camouflage_busy()
        self.assertFalse(page.track_button.isEnabled())
        self.assertFalse(page.learning_button.isEnabled())
        page.start_tracking()               # 버튼을 거치지 않고 불러도 시작하지 않아야 합니다.
        self.assertFalse(page.is_busy())
        window.worker = None
        window.notify_camouflage_busy()
        self.assertTrue(page.track_button.isEnabled())

    def test_main_refuses_to_start_while_camouflage_runs(self):
        window = self.make_window()
        window.set_source(str(ROOT/"sample.mp4"))
        with mock.patch.object(window.camouflage_page, "is_busy", return_value=True), \
                mock.patch("main.TrackingThread") as thread:
            window.play()
        thread.assert_not_called()
        self.assertIsNone(window.worker)
        self.assertIn("보호색 추적 탭", window.statusBar().currentMessage())


if __name__ == "__main__":
    unittest.main()
