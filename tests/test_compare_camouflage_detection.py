"""정답 매칭·검증 분할 제한·결과 보존 검사. 실제 모델 성능 시험은 아닙니다."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

import compare_camouflage_detection as comparison
from extensions.camouflage import CLASS_NAME


class DetectionComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        for folder in ("images/val", "labels/val"):
            (self.data / folder).mkdir(parents=True)
        Image.new("RGB", (100, 100)).save(self.data / "images/val/a.png")
        (self.data / "labels/val/a.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        self.manifest = dict(classes=[CLASS_NAME], counts={"val": {"images": 1}}, records=[
            dict(image="images/val/a.png", split="val", boxes=1),
            dict(image="images/test/DO_NOT_READ.png", split="test", boxes=1)])
        self.save_manifest()
        self.weights = self.root / "camouflage.pt"
        self.weights.write_bytes(b"test-only")
        self.weights.with_suffix(".json").write_text(json.dumps(dict(extension="camouflage", classes=[CLASS_NAME])))

    def save_manifest(self):
        (self.data / "splits.json").write_text(json.dumps(self.manifest))

    def args(self, *extra):
        return comparison.parser().parse_args([
            "--data", str(self.data), "--weights", str(self.weights), "--conf", "0.05", "0.5",
            "--nms", "0.7", "0.3", "--output", str(self.root / "output"), *extra])

    def test_duplicate_counts_as_false_positive_and_confidence_filters_it(self):
        truth = [[30, 30, 70, 70]]
        predictions = [[30, 30, 70, 70, 0.73], [32, 30, 72, 70, 0.21]]
        self.assertEqual(comparison.match_counts(truth, predictions, 0.05),
                         dict(tp=1, fp=1, fn=0, duplicate_candidates=1))
        self.assertEqual(comparison.match_counts(truth, predictions, 0.5),
                         dict(tp=1, fp=0, fn=0, duplicate_candidates=0))

    def test_two_real_objects_and_empty_frame(self):
        truth = [[0, 0, 20, 20], [10, 0, 30, 20]]
        predictions = [[0, 0, 20, 20, 0.9], [10, 0, 30, 20, 0.8]]
        self.assertEqual(comparison.match_counts(truth, predictions, 0.5)["tp"], 2)
        self.assertEqual(comparison.match_counts([], predictions, 0.5)["fp"], 2)
        self.assertEqual(comparison.match_counts(truth, [], 0.5)["fn"], 2)

    def test_one_detection_cannot_match_two_truths(self):
        result = comparison.match_counts([[0, 0, 20, 20]]*2, [[0, 0, 20, 20, 0.8]], 0.5)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertAlmostEqual(comparison.metrics(result)["f1"], 2/3)

    def test_validation_coordinates_and_unread_test_split(self):
        images = comparison.validation_images(self.data)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0][1], [[30, 30, 70, 70]])

    def test_bad_paths_classes_counts_and_labels_are_rejected(self):
        for change in (dict(image="images/test/a.png"), dict(image="../a.png"), dict(boxes=2)):
            with self.subTest(change=change):
                old = dict(self.manifest["records"][0])
                self.manifest["records"][0].update(change)
                self.save_manifest()
                with self.assertRaises(ValueError):
                    comparison.validation_images(self.data)
                self.manifest["records"][0] = old
        self.save_manifest()
        for label in ("1 0.5 0.5 0.4 0.4", "0 nan 0.5 0.4 0.4", "0 0 0 1 1"):
            with self.subTest(label=label):
                (self.data / "labels/val/a.txt").write_text(label)
                with self.assertRaises(ValueError):
                    comparison.validation_images(self.data)

    def test_comparison_predicts_once_per_image_per_nms_and_saves_real_counts(self):
        boxes = SimpleNamespace(xyxy=mock.Mock(), conf=mock.Mock())
        boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[30, 30, 70, 70], [31, 30, 71, 70]])
        boxes.conf.cpu.return_value.numpy.return_value = np.array([0.73, 0.21])
        model = mock.Mock(names={0: CLASS_NAME})
        model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        with mock.patch("ultralytics.YOLO", return_value=model), contextlib.redirect_stdout(io.StringIO()):
            report = comparison.compare(self.args())
        self.assertEqual(model.predict.call_count, 2)
        self.assertEqual([call.kwargs['iou'] for call in model.predict.call_args_list], [0.7, 0.3])
        for call in model.predict.call_args_list:
            self.assertEqual(call.kwargs['conf'], 0.05)
            self.assertEqual(Path(call.args[0]).parent, self.data / 'images/val')
        self.assertEqual(report["best"]["conf"], 0.5)
        self.assertEqual(report["best"]["nms_iou"], 0.7)
        self.assertEqual(report["best"]["f1"], 1)
        self.assertEqual(report["candidates"][0]["duplicate_candidates"], 1)
        self.assertEqual(report["tracking_evaluation"], "미실행")
        self.assertFalse(report["applied"])
        self.assertTrue((self.root / "output/comparison.csv").is_file())
        self.assertEqual(json.loads((self.root / "output/summary.json").read_text())["images"], 1)

    def test_existing_output_bad_grid_hash_and_prediction_failure(self):
        output = self.root / "output"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            comparison.compare(self.args())
        output.rmdir()
        for extra in (("--conf", "nan"), ("--nms", "1")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                comparison.compare(self.args(*extra))
        metadata = self.weights.with_suffix(".json")
        old = metadata.read_text()
        metadata.write_text(json.dumps(dict(json.loads(old), weights_sha256="wrong")))
        with self.assertRaisesRegex(ValueError, "해시"):
            comparison.compare(self.args())
        metadata.write_text(old)
        model = mock.Mock(names={0: CLASS_NAME})
        model.predict.side_effect = RuntimeError("검출 실패")
        with mock.patch("ultralytics.YOLO", return_value=model), self.assertRaisesRegex(RuntimeError, "검출 실패"):
            comparison.compare(self.args())
        self.assertFalse(output.exists())
