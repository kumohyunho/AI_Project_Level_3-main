"""실제 ByteTrack·지표로 튜닝 경로를 검증합니다. 검출기는 가짜이며 실동물 성능 시험이 아닙니다."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import evaluate_camouflage
import run_camouflage
import tune_camouflage as tuning
from extensions.camouflage import CLASS_NAME, default_tracker, read_tracker_config
from tests.test_camouflage import prepare_moca


class FakeDetector:
    names = {0: CLASS_NAME}

    def __init__(self, *args):
        self.calls = []

    def __call__(self, image, threshold):
        self.calls.append(threshold)
        return np.array([[10.0, 10, 30, 40]]), np.array([0.55]), np.array([0])


class TuningTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.data = prepare_moca(self.root)
        self.weights = self.root / "camouflage.pt"
        self.weights.write_bytes(b"fake")
        self.metadata = dict(extension="camouflage", classes=[CLASS_NAME], tracker=default_tracker().to_dict())
        self.weights.with_suffix(".json").write_text(json.dumps(self.metadata))
        self.detector = FakeDetector()
        self.extension = SimpleNamespace(detector=self.detector, close=mock.Mock())

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, split="val", extra=()):
        return tuning.parser().parse_args([
            str(self.data / "tracking" / split), "--weights", str(self.weights),
            "--output", str(self.root / "tuning"), "--high", "0.4", "0.6", "--match", "0.8", "--buffer", "60", *extra])

    def run_tuning(self, args=None):
        with mock.patch("tune_camouflage.make_extension", return_value=self.extension), \
                contextlib.redirect_stdout(io.StringIO()):
            return tuning.tune(args or self.args())

    def test_same_detections_reused_and_existing_setting_kept_on_tie(self):
        result = self.run_tuning()
        self.assertEqual(len(self.detector.calls), 4)
        self.assertEqual(set(self.detector.calls), {0.05})
        self.extension.close.assert_called_once()
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["best"]["candidate"], 0)
        self.assertFalse(result["changed"])
        self.assertEqual(result["best"]["metrics"]["HOTA"], 100)
        bad = next(r for r in result["candidates"] if r["tracker"]["high_thresh"] == 0.6)
        self.assertEqual(bad["metrics"]["FN"], 2)
        self.assertEqual(bad["metrics"]["IDs"], 0)
        for key in ("comparison.csv", "comparison.md", "summary.json", "best_config.json"):
            self.assertTrue((self.root / "tuning" / key).is_file())
        self.assertEqual(len(result["weights_sha256"]), 64)
        self.assertEqual(result["sequences"][0]["evaluated_frames"], [1, 4])
        self.assertEqual(len(result["sequences"][0]["decoded_images_sha256"]), 64)
        self.assertNotIn("fps", result["best"]["metrics"])

    def test_better_candidate_selected_when_baseline_misses_objects(self):
        self.metadata["tracker"]["high_thresh"] = 0.6
        self.metadata["tracker"]["new_track_thresh"] = 0.7
        self.weights.with_suffix(".json").write_text(json.dumps(self.metadata))
        result = self.run_tuning()
        self.assertTrue(result["changed"])
        self.assertEqual(result["best"]["tracker"]["high_thresh"], 0.4)
        self.assertEqual(result["baseline"]["metrics"]["HOTA"], 0)
        self.assertGreater(result["delta"]["HOTA"], 0)

    def test_test_and_train_partitions_rejected_before_model_load(self):
        for split in ("test", "train"):
            with self.subTest(split=split), mock.patch("tune_camouflage.make_extension") as factory:
                with self.assertRaisesRegex(ValueError, "val 분할"):
                    tuning.tune(self.args(split))
                factory.assert_not_called()
                self.assertFalse((self.root / "tuning").exists())

    def test_missing_ids_and_bad_frame_list_rejected(self):
        path = next((self.data / "tracking" / "val").glob("*/annotations.json"))
        info = json.loads(path.read_text())
        for replacement in (dict(ids_verified=False), dict(evaluated_frames=[1.5]), dict(evaluated_frames=[1, 1])):
            path.write_text(json.dumps(dict(info, **replacement)))
            with self.subTest(replacement=replacement), mock.patch("tune_camouflage.make_extension") as factory:
                with self.assertRaises(ValueError):
                    tuning.tune(self.args())
                factory.assert_not_called()

    def test_empty_gt_does_not_produce_a_best_configuration(self):
        path = next((self.data / "tracking" / "val").glob("*/gt/gt.txt"))
        path.write_text("")
        with self.assertRaisesRegex(ValueError, "객체가 없어"):
            self.run_tuning()
        self.assertFalse((self.root / "tuning").exists())

    def test_fractional_or_duplicate_ground_truth_id_rejected(self):
        path = next((self.data / "tracking" / "val").glob("*/gt/gt.txt"))
        original = path.read_text()
        for text in (original.replace(",7,", ",7.5,"), original+original):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.run_tuning()
        self.assertFalse((self.root / "tuning").exists())

    def test_invalid_grid_and_duplicate_candidates(self):
        base = default_tracker()
        for high, match, buffer in (([float("nan")], [0.8], [60]), ([0.01], [0.8], [60]),
                                     ([0.4], [1.2], [60]), ([0.4], [0.8], [0]), ([], [0.8], [60])):
            with self.subTest(high=high, match=match, buffer=buffer), self.assertRaises(ValueError):
                tuning.candidates(base, high, match, buffer)
        self.assertEqual(len(tuning.candidates(base, [0.4, 0.4], [0.8], [60])), 1)
        with self.assertRaises(ValueError):
            tuning.candidates(base, [0.4]*256, [0.8], [60])

    def test_cached_detections_unchanged_and_tracker_reset_for_each_candidate(self):
        sequences = tuning.find_sequences([self.data / "tracking" / "val"])
        annotations = tuning.load_evaluation_annotations(sequences, "val")
        truths = {s.name:tuning.ground_truth_for_tuning(s, annotations[s.name]["evaluated_frames"]) for s in sequences}
        cached = tuning.cache_detections(sequences, annotations, truths, self.detector, 0.05, False)
        before = copy.deepcopy(cached[0].detections)
        first = tuning.score_config(default_tracker(), cached)
        second = tuning.score_config(default_tracker(), cached)
        self.assertEqual(first, second)
        for old, new in zip(before, cached[0].detections):
            for a, b in zip(old[:3], new[:3]):
                np.testing.assert_array_equal(a, b)

    def test_tuning_config_is_reusable_by_evaluator_and_video_runner(self):
        result = self.run_tuning()
        config_path = self.root / "tuning" / "best_config.json"
        self.assertEqual(read_tracker_config(config_path), result["best"]["tracker"])
        args = evaluate_camouflage.parser().parse_args([
            str(self.data / "tracking" / "val"), "--weights", str(self.weights),
            "--tracker-config", str(config_path), "--output", str(self.root / "evaluation")])
        with mock.patch("pipeline.Detector", FakeDetector), contextlib.redirect_stdout(io.StringIO()):
            evaluation = evaluate_camouflage.evaluate(args)
        for metric in tuning.METRICS:
            self.assertEqual(result["best"]["metrics"][metric], evaluation["metrics"][metric])
        parsed = run_camouflage.parser().parse_args(["--video", "new.mp4", "--tracker-config", str(config_path)])
        self.assertEqual(parsed.tracker_config, config_path)

    def test_coordinate_origin_matches_mot_file_evaluation(self):
        gt_path = next((self.data / "tracking" / "val").glob("*/gt/gt.txt"))
        gt_path.write_text(gt_path.read_text().replace(",11.0000,11.0000,", ",1.0000,1.0000,"))
        self.detector = mock.Mock(return_value=(np.array([[0., 0, 20, 30]]), np.array([0.55]), np.array([0])))
        self.extension.detector = self.detector
        result = self.run_tuning()
        self.assertEqual(result["best"]["metrics"]["HOTA"], 100)

    def test_existing_output_and_failed_detection_leave_no_new_result(self):
        output = self.root / "tuning"
        output.mkdir()
        (output / "keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_tuning()
        self.assertEqual((output / "keep.txt").read_text(), "keep")
        (output / "keep.txt").unlink()
        output.rmdir()
        self.extension.detector = mock.Mock(side_effect=RuntimeError("테스트용 검출 오류"))
        with self.assertRaisesRegex(RuntimeError, "테스트용 검출 오류"):
            self.run_tuning()
        self.extension.close.assert_called_once()
        self.assertFalse(output.exists())

    def test_invalid_detector_output_rejected_without_results(self):
        self.extension.detector = mock.Mock(return_value=(np.zeros((1, 4)), np.array([float("nan")]), np.array([0])))
        with self.assertRaisesRegex(ValueError, "검출 결과"):
            self.run_tuning()
        self.assertFalse((self.root / "tuning").exists())

    def test_selected_settings_reject_different_weights(self):
        self.run_tuning()
        config = self.root / "tuning" / "best_config.json"
        other = self.root / "other.pt"
        other.write_bytes(b"different-model")
        with self.assertRaisesRegex(ValueError, "가중치가 다릅니다"):
            read_tracker_config(config, other)
        self.assertEqual(read_tracker_config(config, self.weights)["high_thresh"], 0.4)

    def test_invalid_fps_and_stale_model_metadata_rejected_before_detection(self):
        path = next((self.data / "tracking" / "val").glob("*/seqinfo.ini"))
        original = path.read_text()
        path.write_text(original.replace("frameRate=24.0", "frameRate=-1"))
        with mock.patch("tune_camouflage.make_extension") as factory:
            with self.assertRaisesRegex(ValueError, "FPS"):
                tuning.tune(self.args())
            factory.assert_not_called()
        path.write_text(original)
        self.metadata["weights_sha256"] = "0"*64
        self.weights.with_suffix(".json").write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(ValueError, "가중치 해시"):
            self.run_tuning()


if __name__ == "__main__":
    unittest.main()
