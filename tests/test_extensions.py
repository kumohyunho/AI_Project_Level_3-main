"""확장 기능 틀(extensions/), 결과 파일의 source·label 열 테스트.  실행: ``python -m unittest discover -s tests -t .``"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import extensions  # noqa: E402
from extensions.base import DetectTrackExtension  # noqa: E402
from pipeline import PipelineConfig, TrackRow  # noqa: E402
from results_io import COLUMNS, ResultWriter, load_results  # noqa: E402


class FakeDetector:
    """오른쪽으로 걷는 객체 하나를 매 프레임 같은 점수로 돌려주는 가짜 검출기."""

    def __init__(self, weights, classes, imgsz=640, device="cpu"):
        self.names = {0: classes[0]}
        self.frame = 0

    def __call__(self, frame, conf):
        self.frame += 1
        x = 10.0 * self.frame
        return np.array([[x, 20.0, x + 40.0, 100.0]]), np.array([0.9]), np.array([0])


class RegistryTest(unittest.TestCase):
    def test_registered_extensions_are_listed_with_reason(self):
        items = {key: (label, reason) for key, label, reason in extensions.available_extensions()}
        self.assertEqual(set(items), {"hangul", "camouflage"})
        for key, (label, reason) in items.items():
            cls = extensions.load(key)
            self.assertEqual(cls.key, key)
            self.assertTrue(cls.label and cls.tag)

    def test_unknown_extension(self):
        with self.assertRaises(ValueError):
            extensions.load("nope")

    def test_pipeline_config_keeps_extensions(self):
        config = PipelineConfig(weights="yolo11n.pt", extensions=["camouflage"])
        self.assertEqual(PipelineConfig.from_dict(config.to_dict()).extensions, ["camouflage"])
        self.assertEqual(PipelineConfig.from_dict({"weights": "yolo11n.pt"}).extensions, [])


class DetectTrackExtensionTest(unittest.TestCase):
    def test_tracks_and_labels(self):
        weights = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        weights.close()

        class Dummy(DetectTrackExtension):
            key, label, tag = "dummy", "테스트", "테"
            classes = ["thing"]
            tracker = {"preset": "baseline", "new_track_thresh": 0.5}

            @classmethod
            def weights_path(cls):
                return Path(weights.name)

            def describe(self, frame, row):
                return f"frame{len(row.trail)}"

        with mock.patch("pipeline.Detector", FakeDetector):
            extension = Dummy(fps=30)
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        rows = [extension.process(frame, index) for index in range(1, 5)]
        Path(weights.name).unlink()
        self.assertEqual(extension.tracker_config.new_track_thresh, 0.5)
        visible = [r for frame_rows in rows for r in frame_rows]
        self.assertTrue(visible)
        self.assertEqual({r.track_id for r in visible}, {1})
        self.assertTrue(all(r.source == "dummy" and r.cls_name == "thing" for r in visible))
        self.assertTrue(visible[-1].label.startswith("frame"))
        self.assertEqual(visible[-1].direction, "→")

    def test_missing_weights_raises(self):
        class Missing(DetectTrackExtension):
            key, label, tag, weights = "missing", "없음", "없", "no_such_file.pt"

        self.assertIn("가중치가 없습니다", Missing.check())
        with self.assertRaises(RuntimeError):
            Missing()


class ExtensionOnlyPipelineTest(unittest.TestCase):
    """추적 대상(사람)을 끄고 확장 기능만 켜면 기본 검출기 없이 확장 기능만 실행합니다."""

    def test_extension_only_skips_main_detector(self):
        from pipeline import TrackingPipeline

        class OnlyExtension:
            def __init__(self, device="cpu", fps=30.0):
                self.calls = 0

            def process(self, frame, frame_index):
                self.calls += 1
                return [TrackRow(1001, "단어", (1.0, 2.0, 3.0, 4.0), 1.0, "추적 중", "—", 0.0, 0.0, (),
                                 source="hangul", label="한글")]

            def close(self):
                pass

        detector = mock.Mock(side_effect=AssertionError("기본 검출기를 부르면 안 됩니다"))
        with mock.patch("pipeline.Detector", detector), \
                mock.patch("extensions.load", return_value=OnlyExtension):
            pipeline = TrackingPipeline(PipelineConfig(classes=[], extensions=["hangul"], mosaic=False))
            rows, waiting, timings = pipeline.process(np.zeros((60, 80, 3), dtype=np.uint8))
        detector.assert_not_called()
        self.assertEqual([(r.source, r.label) for r in rows], [("hangul", "한글")])
        self.assertEqual(waiting, [])
        self.assertEqual((timings["detect"], timings["track"]), (0.0, 0.0))
        self.assertEqual(pipeline.extensions[0].calls, 1)

    def test_nothing_selected_raises(self):
        from pipeline import TrackingPipeline

        with mock.patch("pipeline.Detector"), self.assertRaises(ValueError):
            TrackingPipeline(PipelineConfig(classes=[], extensions=[], mosaic=False))


class ResultFileTest(unittest.TestCase):
    def test_source_and_label_round_trip(self):
        writer = ResultWriter()
        writer.add(1, 0, [TrackRow(1, "person", (0, 0, 10, 20), 0.9, "추적 중"),
                          TrackRow(1, "text", (5, 5, 50, 20), 0.8, "추적 중", source="hangul", label="안녕")])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.csv"
            writer.save(path, {"source": "x"})
            frames, _ = load_results(path)
        main_row, text_row = frames[1]
        self.assertEqual((main_row["source"], main_row["label"]), ("main", ""))
        self.assertEqual((text_row["source"], text_row["label"]), ("hangul", "안녕"))

    def test_old_csv_without_source_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.csv"
            path.write_text(",".join(COLUMNS) + "\n1,0,3,person,0,0,10,20,0.9,추적 중,→,1.0,2.0\n",
                            encoding="utf-8-sig")
            frames, _ = load_results(path)
        self.assertEqual(frames[1][0]["source"], "main")
        self.assertEqual(frames[1][0]["label"], "")


if __name__ == "__main__":
    unittest.main()
