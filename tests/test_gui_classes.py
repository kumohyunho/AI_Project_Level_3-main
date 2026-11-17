"""실시간 추적 탭의 추적 대상이 사람 하나뿐인지 검사합니다.

기본 모델(`models/person.pt`)은 사람 전용이라, 고양이·개·새를 고르면 시작할 때 오류가 났습니다.
그래서 고양이·개·새 선택을 화면에서 뺐습니다. 예전에 저장한 설정 파일에 고양이 등이 있어도 사람만 불러옵니다.
"""
import json
import os
import tempfile
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
class GuiClassesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self):
        with mock.patch("main.QMediaPlayer"), mock.patch("main.QAudioOutput"):
            window = main.Window()
        self.addCleanup(window.close)
        return window

    def test_only_person_can_be_selected(self):
        window = self.make_window()
        self.assertEqual(list(window.class_checks), ["person"])
        self.assertEqual(window.pipeline_config().classes, ["person"])

    def test_old_settings_with_animals_load_person_only(self):
        window = self.make_window()
        config = window.pipeline_config().to_dict()
        config["classes"] = ["person", "cat", "dog", "bird"]
        data = {"version": 2, "mode": main.MODE_TRACK, "pipeline": config,
                "display": {key: check.isChecked() for key, check in window.options.items()}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with mock.patch("main.QFileDialog.getOpenFileName", return_value=(str(path), "")), \
                    mock.patch("main.QMessageBox.warning") as warning:
                window.load_settings()
        warning.assert_not_called()
        self.assertEqual(window.pipeline_config().classes, ["person"])


if __name__ == "__main__":
    unittest.main()
