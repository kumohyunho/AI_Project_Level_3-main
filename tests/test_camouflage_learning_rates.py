"""학습률 실험 조건·분할·보존 검사. 실제 학습 성능을 측정하지 않습니다."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

import compare_camouflage_learning_rates as comparison
import train_camouflage
from extensions.camouflage import CLASS_NAME, default_tracker
from pipeline import TrackRow
from results_io import ResultWriter


class LearningRatesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "dataset"
        records = []
        for index, split in enumerate(("train", "val")):
            image = self.data / f"images/{split}/a.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (80, 60), (index*100, 40, 20)).save(image)
            label = self.data / f"labels/{split}/a.txt"
            label.parent.mkdir(parents=True)
            label.write_text("0 0.5 0.5 0.5 0.5\n")
            records.append(dict(image=str(image.relative_to(self.data)), split=split, boxes=1))
        # test 데이터는 실제로 만들지 않습니다. 탐색에서 읽으면 실패해야 합니다.
        records.append(dict(image="images/test/missing.png", split="test", boxes=1))
        self.manifest = dict(classes=[CLASS_NAME], records=records,
                             counts={"train": {"images": 1}, "val": {"images": 1}, "test": {"images": 0}})
        (self.data/"splits.json").write_text(json.dumps(self.manifest))
        (self.data/"dataset.yaml").write_text("path: /content/old_colab_location\n")
        self.weights = self.root / "initial.pt"
        self.weights.write_bytes(b"initial weights")
        self.video = self.root / "input.mp4"
        self.video.write_bytes(b"mock input video")
        self.calls = []

    def args(self, *extra):
        return comparison.parser().parse_args([
            "--data", str(self.data/"dataset.yaml"), "--weights", str(self.weights),
            "--video", str(self.video), "--output", str(self.root/"result"), *extra])

    def fake_train(self, args):
        self.calls.append(args)
        checkpoint = args.runs / "weights/best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"trained at {args.lr0}".encode())
        return dict(extension="camouflage", classes=[CLASS_NAME], best=str(checkpoint),
                    tracker=default_tracker().to_dict(), scores={"val": {"mAP50": 0.6, "mAP50-95": 0.4+args.lr0*100}})

    @staticmethod
    def fake_video(args):
        args.output.mkdir(parents=True)
        writer = ResultWriter()
        writer.add(1, 0, [TrackRow(1, CLASS_NAME, (0, 0, 20, 20), 0.8, "추적 중", source="camouflage")])
        writer.save(args.output/"tracks.csv", dict(frames=2))
        (args.output/"tracked.mp4").write_bytes(b"not a real video")

    def run_job(self, args=None):
        with mock.patch("train_camouflage.train", side_effect=self.fake_train), \
                mock.patch("run_camouflage.run", side_effect=self.fake_video), contextlib.redirect_stdout(io.StringIO()):
            return comparison.compare(args or self.args())

    def test_same_initial_weights_conditions_val_only_and_portable_outputs(self):
        report = self.run_job()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["best_candidate"], 1)
        self.assertEqual(len(self.calls), 2)
        for call in self.calls:
            self.assertEqual(call.weights, self.weights)
            self.assertTrue(call.val_only)
            self.assertEqual(call.patience, 0)
            self.assertEqual(call.seed, 42)
            self.assertFalse(call.install)
        self.assertEqual([call.lr0 for call in self.calls], [0.0001, 0.0005])
        config = json.loads((self.root/"result/dataset_config/dataset.yaml").read_text())
        self.assertEqual(config["path"], str(self.data))
        self.assertNotIn("test", config)
        self.assertTrue((self.root/"result/comparison.csv").is_file())
        for row in report["candidates"]:
            self.assertFalse(Path(row["video"]).is_absolute())
            self.assertTrue((self.root/"result"/row["weights"]).is_file())
        self.assertEqual(self.weights.read_bytes(), b"initial weights")

    def test_bad_rates_and_missing_training_files_fail_before_training(self):
        for extra in (("--lr", "0.1"), ("--lr", "0.1", "0.1"), ("--lr", "nan", "0.1"),
                      ("--epochs", "0"), ("--frames", "0")):
            with self.subTest(extra=extra), mock.patch("train_camouflage.train") as train, self.assertRaises(ValueError):
                comparison.compare(self.args(*extra))
            train.assert_not_called()
        (self.data/"images/train/a.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_job()
        self.assertFalse((self.root/"result").exists())

    def test_cross_split_duplicates_and_bad_labels_rejected(self):
        training = self.data/"images/train/a.png"
        original = training.read_bytes()
        training.write_bytes((self.data/"images/val/a.png").read_bytes())
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "중복"):
            comparison.prepare(self.args())
        training.write_bytes(original)
        (self.data/"labels/train/a.txt").write_text("0 nan 0.5 0.5 0.5\n")
        with self.assertRaisesRegex(ValueError, "형식"):
            comparison.prepare(self.args())

    def test_existing_results_preserved_and_failed_job_not_completed(self):
        output = self.root/"result"
        output.mkdir()
        (output/"keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_job()
        self.assertEqual((output/"keep.txt").read_text(), "keep")
        (output/"keep.txt").unlink()
        output.rmdir()
        with mock.patch("train_camouflage.train", side_effect=RuntimeError("학습 실패")), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "학습 실패"):
            comparison.compare(self.args())
        report = json.loads((output/"comparison.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["candidates"], [])

    def test_train_forwards_lr_and_evaluates_val_only(self):
        checkpoint = self.root/"best.pt"
        checkpoint.write_bytes(b"fake checkpoint")
        calls = []
        class FakeYOLO:
            names = {0: CLASS_NAME}
            trainer = SimpleNamespace(best=checkpoint)
            def __init__(self, path):
                pass
            def train(self, **kwargs):
                calls.append(kwargs)
            def val(self, **kwargs):
                calls.append(kwargs["split"])
                return SimpleNamespace(box=SimpleNamespace(map50=0.6, map=0.4))
        args = train_camouflage.parser().parse_args([
            "--data", str(self.data/"dataset.yaml"), "--weights", str(self.weights), "--lr0", "0.0001", "--val-only"])
        with mock.patch("ultralytics.YOLO", FakeYOLO), contextlib.redirect_stdout(io.StringIO()):
            metadata = train_camouflage.train(args)
        self.assertEqual(calls[0]["lr0"], 0.0001)
        self.assertEqual(calls[1:], ["val"])
        self.assertEqual(metadata["evaluation_splits"], ["val"])
