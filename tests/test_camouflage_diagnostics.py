"""실제 추적기와 가짜 검출기로 진단의 정확한 관찰·저장·결과 불변성을 검증합니다."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from camouflage_diagnostics import DiagnosticRecorder
from extensions.camouflage import CLASS_NAME, default_tracker
from pipeline import TrackRow
from tracker import ByteTracker
import run_camouflage


class DiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def update(self, tracker, recorder, frame, boxes, scores):
        trace = {}
        tracks = tracker.update(np.asarray(boxes).reshape(-1,4), scores, [0]*len(boxes), diagnostics=trace)
        rows = [TrackRow(t.track_id, CLASS_NAME, tuple(t.box), t.score, t.status) for t in tracks]
        recorder.add(frame, trace, rows)
        return trace, rows

    def test_empty_frames_reappearance_and_lost_predictions_are_separate(self):
        tracker = ByteTracker(default_tracker())
        recorder = DiagnosticRecorder(self.root, dict(test="synthetic"))
        self.update(tracker, recorder, 1, [[10,10,50,60]], [0.9])
        self.update(tracker, recorder, 2, [], [])
        self.update(tracker, recorder, 3, [[10,10,50,60]], [0.9])
        recorder.finish("completed")
        frames = [json.loads(line) for line in (self.root/"diagnostic_frames.jsonl").read_text().splitlines()]
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[1]["detections"], [])
        self.assertEqual(frames[1]["outputs"], [])
        self.assertEqual(frames[1]["states"][0]["state"], "lost")
        self.assertEqual(frames[0]["associations"][0]["stage"], "new")
        report = json.loads((self.root/"diagnostic_summary.json").read_text())
        self.assertEqual(report["total_visible_track_rows"], 2)
        self.assertEqual(report["event_counts"]["tracker_reappearance"], 1)
        self.assertEqual(report["event_counts"]["output_gap_started"], 1)
        self.assertEqual(report["tracking_accuracy"], "미측정")
        self.assertNotIn("IDSW", report)

    def test_replacement_is_candidate_even_before_new_id_confirmation(self):
        config = default_tracker()
        config.track_buffer = 1
        tracker = ByteTracker(config)
        recorder = DiagnosticRecorder(self.root, {})
        for frame, score in enumerate((0.9, None, None, 0.9, 0.9), 1):
            self.update(tracker, recorder, frame, [[10,10,50,60]] if score else [], [score] if score else [])
        recorder.finish("completed")
        report = json.loads((self.root/"diagnostic_summary.json").read_text())
        event = report["examples"]["id_replacement_candidate"][0]
        self.assertEqual(event["details"]["old_id"], 1)
        self.assertEqual(event["details"]["new_id"], 2)
        self.assertFalse(event["details"]["new_visible"])
        self.assertIn("확정 아님", event["description"])

    def test_overlap_counts_are_candidates_and_examples_are_bounded(self):
        tracker = ByteTracker(default_tracker())
        recorder = DiagnosticRecorder(self.root, {})
        self.update(tracker, recorder, 1, [[10,10,50,60], [12,10,52,60]], [0.9,0.9])
        for frame in range(2, 32):
            self.update(tracker, recorder, frame, [], [])
        recorder.finish("completed")
        report = json.loads((self.root/"diagnostic_summary.json").read_text())
        self.assertEqual(report["event_counts"]["detection_overlap"], 1)
        self.assertEqual(report["event_counts"]["track_overlap"], 1)
        self.assertEqual(report["event_counts"]["empty_detections"], 30)
        self.assertEqual(len(report["examples"]["empty_detections"]), 20)
        self.assertEqual(report["processed_frames"], 31)

    def test_existing_files_failed_status_and_bad_frame_order(self):
        recorder = DiagnosticRecorder(self.root, {})
        with self.assertRaises(FileExistsError):
            DiagnosticRecorder(self.root, {})
        with self.assertRaises(ValueError):
            recorder.add(3, dict(tracker_frame=3), [])
        recorder.finish("failed", dict(error="synthetic failure"))
        report = json.loads((self.root/"diagnostic_summary.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["processed_frames"], 0)
        self.assertTrue((self.root/"diagnostic_report.html").is_file())

    def model(self):
        weights = self.root/"camouflage.pt"
        weights.write_bytes(b"fake weights")
        weights.with_suffix(".json").write_text(json.dumps(dict(extension="camouflage", classes=[CLASS_NAME], tracker=default_tracker().to_dict())))
        return weights

    def test_extension_records_without_extra_detector_calls_or_changed_tracks(self):
        class Detector:
            names = {0: CLASS_NAME}
            def __init__(self, *args):
                self.calls = 0
            def __call__(self, frame, conf):
                self.calls += 1
                return np.array([[10.,10,50,60]]), np.array([0.15 if self.calls == 2 else 0.9]), np.array([0])
        weights = self.model()
        with mock.patch("pipeline.Detector", Detector):
            before = run_camouflage.make_extension(weights)
            after = run_camouflage.make_extension(weights)
        recorder = DiagnosticRecorder(self.root/"diagnostic", {})
        after.enable_diagnostics(recorder)
        image = np.zeros((80,80,3), dtype=np.uint8)
        original = image.copy()
        for frame in range(1,4):
            a, b = before.process(image, frame), after.process(image, frame)
            self.assertEqual([r.track_id for r in a], [r.track_id for r in b])
            np.testing.assert_array_equal(a[0].box, b[0].box)
        self.assertEqual(after.detector.calls, 3)
        np.testing.assert_array_equal(image, original)
        recorder.finish("completed")

    def test_video_cli_saves_same_frame_diagnostic_video_and_marks_failure(self):
        import cv2
        class Detector:
            names = {0: CLASS_NAME}
            def __init__(self, *args):
                pass
            def __call__(self, frame, conf):
                return np.array([[10.,10,50,60]]), np.array([0.9]), np.array([0])
        weights = self.model()
        source = self.root/"input.avi"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80,80))
        self.assertTrue(writer.isOpened())
        for _ in range(3):
            writer.write(np.zeros((80,80,3), dtype=np.uint8))
        writer.release()
        args = run_camouflage.parser().parse_args([
            "--video", str(source), "--weights", str(weights), "--diagnostics", "--output", str(self.root/"out")])
        with mock.patch("pipeline.Detector", Detector), contextlib.redirect_stdout(io.StringIO()):
            metadata = run_camouflage.run(args)
        self.assertIn("diagnostics", metadata)
        cap = cv2.VideoCapture(str(self.root/"out/diagnostic.mp4"))
        try:
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 160)
        finally:
            cap.release()
        report = json.loads((self.root/"out/diagnostic_summary.json").read_text())
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(report["metadata"]["source_sha256"]), 64)
        journal = [json.loads(line) for line in (self.root/"out/diagnostic_frames.jsonl").read_text().splitlines()]
        self.assertEqual(journal[1]["time_ms"], 100)
        args.output = self.root/"without_diagnostics"
        args.diagnostics = False
        with mock.patch("pipeline.Detector", Detector), contextlib.redirect_stdout(io.StringIO()):
            run_camouflage.run(args)
        self.assertEqual((args.output/"tracks.csv").read_bytes(), (self.root/"out/tracks.csv").read_bytes())
        self.assertFalse((args.output/"diagnostic_frames.jsonl").exists())
        args.output = self.root/"failed"
        args.diagnostics = True
        with mock.patch("pipeline.Detector", Detector), mock.patch.object(Detector, "__call__", side_effect=RuntimeError("detector failed")), \
                self.assertRaisesRegex(RuntimeError, "detector failed"):
            run_camouflage.run(args)
        report = json.loads((self.root/"failed/diagnostic_summary.json").read_text())
        self.assertEqual(report["status"], "failed")
