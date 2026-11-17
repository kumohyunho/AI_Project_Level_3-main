"""보호색 데이터 미리보기·모델 정보·비동기 영상 처리 UI 검사. 동물 정확도 시험이 아닙니다."""
import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QProcess
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    import camouflage_tracking_page as workspace
    from extensions.camouflage import CLASS_NAME, default_tracker
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, "PySide6 설치가 필요합니다.")
class CamouflagePageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        import cv2
        import numpy as np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/"data"
        records = []
        for index, split in enumerate(("train", "val", "test")):
            path = self.data/f"images/{split}/a.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (80, 60), (30+index*20, 60, 20)).save(path)
            label = self.data/f"labels/{split}/a.txt"
            label.parent.mkdir(parents=True)
            label.write_text("0 0.5 0.5 0.5 0.5\n")
            records.append(dict(image=f"images/{split}/a.png", split=split, source="synthetic", boxes=1))
        self.manifest = dict(classes=["camouflaged_animal"], records=records,
                             counts={split: {"images": 1} for split in ("train", "val", "test")})
        (self.data/"splits.json").write_text(json.dumps(self.manifest))
        (self.data/"dataset.yaml").write_text("path: /content/old-path\n")
        self.model = self.root/"camouflage.pt"
        self.model.write_bytes(b"fake model for UI")
        self.model.with_suffix(".json").write_text(json.dumps(dict(
            extension="camouflage", classes=[CLASS_NAME], tracker=default_tracker().to_dict(),
            weights_sha256=hashlib.sha256(self.model.read_bytes()).hexdigest(),
            train_options=dict(epochs=3, lr0=0.0005), scores={"val": {"mAP50": 0.6}})))
        self.video = self.root/"original.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 60))
        self.assertTrue(writer.isOpened())
        for _ in range(2):
            writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
        writer.release()
        self.root_patch = mock.patch("camouflage_tracking_page.ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.page = workspace.CamouflagePage()
        self.addCleanup(self.page.close)

    def wait_idle(self):
        deadline = time.monotonic()+8
        while self.page.is_busy() and time.monotonic() < deadline:
            QTest.qWait(25)
        QTest.qWait(100)
        self.assertFalse(self.page.is_busy())

    def test_ground_truth_and_split_preview_preserve_original(self):
        original = (self.data/"images/train/a.png").read_bytes()
        root, records = workspace.dataset_records(self.data/"dataset.yaml")
        preview = workspace.ground_truth_preview(root, records[0])
        self.assertEqual(preview.getpixel((20, 15)), (0, 255, 112))
        self.assertEqual((self.data/"images/train/a.png").read_bytes(), original)
        self.page.load_dataset(self.data)
        self.assertEqual(self.page.images.rowCount(), 1)
        self.assertIsNotNone(self.page.preview_image)
        self.page.split.setCurrentIndex(1)
        self.assertEqual(self.page.visible_records[0]["split"], "val")
        self.assertIn("정답 박스", self.page.preview_info.text())

    def test_missing_images_shown_as_missing_and_bad_paths_labels_rejected(self):
        image = self.data/"images/train/a.png"
        image.unlink()
        self.page.load_dataset(self.data)
        self.assertIsNone(self.page.preview_image)
        self.assertIn("파일을 확인", self.page.preview.text())
        self.manifest["records"][0]["image"] = "../outside.png"
        (self.data/"splits.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            workspace.dataset_records(self.data)
        (self.data/"labels/val/a.txt").write_text("0 nan 0.5 0.5 0.5\n")
        with self.assertRaises(ValueError):
            workspace.ground_truth_preview(self.data, self.manifest["records"][1])

    def test_model_metadata_and_mismatched_hash(self):
        self.page.load_model(self.model)
        self.assertIn("60.00%", self.page.model_info.text())
        self.assertIn("재측정 아님", self.page.model_info.text())
        self.assertIn("0.7", self.page.model_info.text())
        self.model.write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            workspace.model_information(self.model)

    def test_video_and_learning_selection_are_forwarded_without_data_copy(self):
        self.page.load_dataset(self.data)
        self.page.load_model(self.model)
        self.page.load_video(self.video)
        self.assertTrue(self.page.track_button.isEnabled())
        self.page.learning_button.click()
        dialog = self.page.learning_dialog
        self.assertEqual(dialog.data.text(), str(self.data/"dataset.yaml"))
        self.assertEqual(dialog.weights.text(), str(self.model))
        self.assertEqual(dialog.video.text(), str(self.video))
        dialog.close()
        self.page.result_video = self.video
        self.page.load_model(self.model)
        self.assertIsNone(self.page.result_video)

    def test_async_tracking_result_playback_and_stop(self):
        script = self.root/"run_camouflage.py"
        script.write_text("""import argparse, pathlib, shutil, json
p=argparse.ArgumentParser()
for key in ('video','weights','device','output'): p.add_argument('--'+key)
p.add_argument('--diagnostics',action='store_true')
a=p.parse_args()
out=pathlib.Path(a.output); out.mkdir(parents=True)
shutil.copyfile(a.video,out/'tracked.mp4')
(out/'tracks.csv').write_text('synthetic UI test')
(out/'tracks.json').write_text(json.dumps(vars(a)))
if a.diagnostics:
    shutil.copyfile(a.video,out/'diagnostic.mp4')
    (out/'diagnostic_summary.json').write_text(json.dumps({'status':'completed'}))
    (out/'diagnostic_report.html').write_text('synthetic diagnostic report')
print('synthetic UI worker complete',flush=True)
""")
        self.page.load_model(self.model)
        self.page.load_video(self.video)
        self.page.start_tracking()
        self.assertFalse(self.page.learning_button.isEnabled())
        self.wait_idle()
        self.assertTrue(self.page.result_video.is_file())
        self.assertTrue(self.page.result_button.isEnabled())
        self.page.set_external_busy(True)
        self.page.set_busy(False)
        self.assertFalse(self.page.learning_button.isEnabled())
        self.page.set_external_busy(False)
        self.assertIn("진단 저장 완료", self.page.status.text())
        self.assertTrue(self.page.diagnostic_button.isEnabled())
        self.assertTrue(self.page.diagnostic_video_button.isEnabled())
        self.page.play_diagnostic()
        self.assertTrue(self.page.player.source().toLocalFile().endswith("diagnostic.mp4"))
        with mock.patch("camouflage_tracking_page.QDesktopServices.openUrl") as opener:
            self.page.open_diagnostics()
            self.assertTrue(opener.call_args.args[0].toLocalFile().endswith("diagnostic_report.html"))
        self.assertIn("synthetic", self.page.log.toPlainText())
        self.page.player.stop()
        script.write_text("import time; print('waiting',flush=True); time.sleep(30)")
        self.page.start_tracking()
        deadline = time.monotonic()+5
        while self.page.process.state() != QProcess.Running and time.monotonic() < deadline:
            QTest.qWait(25)
        self.page.stop_processing()
        self.wait_idle()
        self.assertIsNone(self.page.result_video)
        self.assertFalse(self.page.diagnostic_button.isEnabled())
        self.assertTrue(self.page.cancelled)
        self.assertIn("중지됨", self.page.status.text())

    def test_learning_callback_and_result_without_files_not_promoted(self):
        callback = mock.Mock()
        page = workspace.CamouflagePage(on_learning=callback)
        try:
            page.learning_button.click()
            callback.assert_called_once()
            (self.root/"run_camouflage.py").write_text("print('no output')")
            self.page.load_model(self.model)
            self.page.load_video(self.video)
            self.page.start_tracking()
            self.wait_idle()
            self.assertIsNone(self.page.result_video)
            self.assertFalse(self.page.result_button.isEnabled())
        finally:
            page.close()
