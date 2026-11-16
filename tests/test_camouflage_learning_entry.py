"""메인 GUI의 보호색 학습 비교 진입·추적 중 비활성화 검사."""
import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication
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
            self.assertTrue(window.learning_button.isEnabled())
            window.learning_button.click()
            self.assertTrue(window.learning_dialog.isVisible())
            self.assertTrue(window.learning_dialog.isModal())
            window.learning_dialog.close()
            window.set_settings_enabled(False)
            self.assertFalse(window.learning_button.isEnabled())
            self.app.processEvents()
        finally:
            window.close()
