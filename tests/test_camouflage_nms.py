"""가짜 검출 제안과 실제 NMS·ByteTrack으로 설정 연결을 검사합니다. 동물 성능 시험이 아닙니다."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torchvision.ops import nms

import compare_camouflage_video as video_comparison
import configure_camouflage_inference as configuration
import run_camouflage
from extensions.camouflage import (CLASS_NAME, CamouflageExtension, default_tracker,
                                   read_inference_config, validate_nms)
from pipeline import PipelineConfig, TrackingPipeline


class ProposalDetector:
    """YOLO 신경망 대신 고정 제안을 만들고 torchvision의 실제 NMS를 사용합니다."""
    proposals = [[10., 10., 70., 50.], [30., 10., 80., 50.]]
    low = False
    hidden = False

    def __init__(self, weights, classes, imgsz=640, device="cpu"):
        self.names = {0: classes[0]}
        self.class_ids, self.imgsz, self.device = [0], imgsz, device
        self.model = self
        self.calls = []

    def predict(self, frame, conf, iou=0.7, **kwargs):
        self.calls.append(dict(conf=conf, iou=iou, **kwargs))
        boxes = torch.tensor(self.proposals, dtype=torch.float32).reshape(-1, 4)
        scores = torch.tensor([0.15, 0.12] if self.low else [0.9, 0.8])[:len(boxes)]
        keep = scores >= conf
        if self.hidden:
            keep[:] = False
        boxes, scores = boxes[keep], scores[keep]
        retained = nms(boxes, scores, iou)
        result = SimpleNamespace(xyxy=boxes[retained], conf=scores[retained], cls=torch.zeros(len(retained)))
        return [SimpleNamespace(boxes=result)]

    def __call__(self, frame, conf):
        result = self.predict(frame, conf)[0].boxes
        return result.xyxy.numpy(), result.conf.numpy(), result.cls.numpy().astype(int)


class NmsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root / "camouflage.pt"
        self.weights.write_bytes(b"synthetic model, not real weights")
        self.digest = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        self.weights.with_suffix(".json").write_text(json.dumps(dict(
            extension="camouflage", classes=[CLASS_NAME], weights_sha256=self.digest,
            tracker=default_tracker().to_dict())))

    def install(self, *extra):
        args = configuration.parser().parse_args(["--weights", str(self.weights), *extra])
        with contextlib.redirect_stdout(io.StringIO()):
            return configuration.configure(args)

    def test_configuration_round_trip_preserves_training_files_and_tracker_thresholds(self):
        original = self.weights.with_suffix(".json").read_bytes()
        self.install("--nms-iou", "0.3")
        self.assertEqual(read_inference_config(self.weights), dict(nms_iou=0.3, max_det=300))
        self.assertEqual(self.weights.with_suffix(".json").read_bytes(), original)
        self.assertEqual(hashlib.sha256(self.weights.read_bytes()).hexdigest(), self.digest)
        with mock.patch("pipeline.Detector", ProposalDetector):
            extension = run_camouflage.make_extension(self.weights)
        self.assertEqual(extension.tracker_config.low_thresh, 0.05)
        self.assertEqual(extension.tracker_config.high_thresh, 0.4)
        self.assertEqual(extension.tracker_config.new_track_thresh, 0.5)
        with self.assertRaises(FileExistsError):
            self.install("--nms-iou", "0.5")

    def test_actual_nms_reduces_overlapping_proposals_before_real_tracker(self):
        frame = np.zeros((60, 100, 3), dtype=np.uint8)
        with mock.patch("pipeline.Detector", ProposalDetector):
            before = run_camouflage.make_extension(self.weights, nms_iou=0.7)
            after = run_camouflage.make_extension(self.weights, nms_iou=0.3)
        self.assertEqual(len(before.process(frame, 1)), 2)
        self.assertEqual(len(after.process(frame, 1)), 1)
        call = after.detector.base.calls[0]
        self.assertEqual(call["iou"], 0.3)
        self.assertEqual(call["conf"], 0.05)
        self.assertEqual(call["max_det"], 300)

    def test_two_separated_animals_are_not_forced_to_one(self):
        with mock.patch("pipeline.Detector", ProposalDetector):
            extension = run_camouflage.make_extension(self.weights, nms_iou=0.3)
        extension.detector.base.proposals = [[0., 0., 20., 40.], [60., 0., 80., 40.]]
        self.assertEqual(len(extension.process(np.zeros((60, 100, 3), dtype=np.uint8), 1)), 2)

    def test_low_score_identity_and_occlusion_reentry_survive_configuration(self):
        self.install("--nms-iou", "0.3")
        with mock.patch("pipeline.Detector", ProposalDetector):
            extension = run_camouflage.make_extension(self.weights)
        frame = np.zeros((60, 100, 3), dtype=np.uint8)
        first = extension.process(frame, 1)
        base = extension.detector.base
        base.low = True
        low = extension.process(frame, 2)
        self.assertEqual([r.track_id for r in first], [r.track_id for r in low])
        self.assertEqual(len(low), 1)
        self.assertAlmostEqual(low[0].score, 0.15)
        base.hidden = True
        self.assertEqual(extension.process(frame, 3), [])
        base.hidden, base.low = False, False
        returned = extension.process(frame, 4)
        self.assertEqual([r.track_id for r in returned], [r.track_id for r in first])

    def test_gui_pipeline_loads_same_model_sidecar(self):
        self.install("--nms-iou", "0.3")
        with mock.patch("pipeline.Detector", ProposalDetector), \
                mock.patch.object(CamouflageExtension, "weights_path", return_value=self.weights):
            pipeline = TrackingPipeline(PipelineConfig(weights="fake", mosaic=False, extensions=["camouflage"]))
        try:
            rows, _, _ = pipeline.process(np.zeros((60, 100, 3), dtype=np.uint8))
            self.assertEqual(len([row for row in rows if row.source == "camouflage"]), 1)
            self.assertEqual(len([row for row in rows if row.source == "main"]), 2)
        finally:
            pipeline.close()

    def test_report_installs_only_nms_and_rejects_different_model_split_or_resolution(self):
        best = dict(conf=0.2, nms_iou=0.3, f1=0.6)
        report = dict(split="val", imgsz=640, max_det=300, weights_sha256=self.digest, best=best, candidates=[best])
        path = self.root / "summary.json"
        for change in (dict(split="test"), dict(weights_sha256="bad"), dict(imgsz=320), dict(candidates=[])):
            path.write_text(json.dumps(dict(report, **change)))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.install("--report", str(path))
        path.write_text(json.dumps(report))
        installed = self.install("--report", str(path))
        self.assertEqual(installed["nms_iou"], 0.3)
        self.assertFalse(installed["selected_conf_applied"])

    def test_invalid_nms_and_wrong_model_are_blocked_before_model_load(self):
        for value in (float("nan"), float("inf"), 0, 1, True, "0.3", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_nms(value)
        self.install("--nms-iou", "0.3")
        settings = self.weights.with_suffix(".inference.json")
        original = settings.read_text()
        settings.write_text(json.dumps(dict(json.loads(original), max_det=500)))
        with self.assertRaisesRegex(ValueError, "max_det"):
            read_inference_config(self.weights)
        settings.write_text(original)
        self.weights.write_bytes(b"changed model")
        with self.assertRaisesRegex(ValueError, "해시"):
            read_inference_config(self.weights)
        with mock.patch.object(CamouflageExtension, "weights_path", return_value=self.weights):
            self.assertIn("해시", CamouflageExtension.check())

    def test_video_comparison_outputs_both_videos_and_diagnostics_atomically(self):
        import cv2

        video = self.root / "source.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 60))
        self.assertTrue(writer.isOpened())
        for _ in range(4):
            writer.write(np.zeros((60, 100, 3), dtype=np.uint8))
        writer.release()
        args = video_comparison.parser().parse_args([
            "--video", str(video), "--weights", str(self.weights), "--output", str(self.root/"compare")])
        with mock.patch("pipeline.Detector", ProposalDetector), contextlib.redirect_stdout(io.StringIO()):
            report = video_comparison.compare(args)
        self.assertEqual(report["before"]["diagnostics"]["unique_ids"], 2)
        self.assertEqual(report["after"]["diagnostics"]["unique_ids"], 1)
        self.assertEqual(report["before"]["diagnostics"]["overlap_frames"], 4)
        self.assertEqual(report["after"]["diagnostics"]["overlap_frames"], 0)
        for name in ("before", "after"):
            self.assertTrue((self.root/"compare"/name/"tracked.mp4").is_file())
        self.assertTrue((self.root/"compare/comparison.csv").is_file())
        self.assertTrue((self.root/"compare/comparison.json").is_file())
        self.assertIn("미실행", report["tracking_evaluation"])
        with self.assertRaises(FileExistsError):
            video_comparison.compare(args)
        args.output = self.root / "failed"
        with mock.patch("run_camouflage.run", side_effect=RuntimeError("테스트 실패")), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            video_comparison.compare(args)
        self.assertFalse(args.output.exists())
