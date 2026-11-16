"""보호색 확장 검증: 실제 데이터·신경망 가중치 없이 실행합니다."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

import camouflage_dataset as dataset
import evaluate_camouflage
import run_camouflage
import train_camouflage
from extensions.camouflage import CLASS_NAME, CamouflageExtension, default_tracker
from pipeline import TrackRow
from results_io import load_results


def image(path, value=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 60), (value, value // 2, 20)).save(path)


def moca_fixture(root, ids=True):
    rows = []
    for index, name in enumerate(("fox", "crab", "moth")):
        for frame in range(4):
            image(root / "frames" / name / f"{frame:05d}.png", 60 + index * 30 + frame)
        for frame in (0, 3):
            rows.append(dict(sequence=name, filename=f"{frame:05d}.png", x=10, y=10,
                             width=20, height=30, track_id=7 if ids else ""))
    path = root / "boxes.csv"
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def prepare_moca(root, ids=True, single=False):
    annotations = moca_fixture(root, ids)
    args = dataset.parser().parse_args([
        "--moca-frames", str(root / "frames"), "--moca-csv", str(annotations),
        "--output", str(root / "prepared"), *(["--single-object"] if single else [])])
    dataset.prepare(args)
    return root / "prepared"


class DatasetTest(unittest.TestCase):
    def test_binary_and_instance_masks_do_not_split_disconnected_body(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mask.png"
            mask = np.zeros((20, 30), np.uint8)
            mask[2:5, 3:7] = 1
            mask[10:15, 20:25] = 1
            Image.fromarray(mask).save(path)
            self.assertEqual(dataset.mask_boxes(path)[0], [(3, 2, 25, 15)])
            self.assertEqual(dataset.mask_boxes(path, True)[0], [(3, 2, 25, 15)])
            mask[10:15, 20:25] = 2
            Image.fromarray(mask).save(path)
            self.assertEqual(dataset.mask_boxes(path, True)[0], [(3, 2, 7, 5), (20, 10, 25, 15)])

    def test_cod10k_split_background_and_noncam(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for official, count in (("Train", 8), ("Test", 2)):
                for n in range(count):
                    name = f"COD10K-CAM-{official}-{n}"
                    image(root / official / "Image" / f"{name}.png", 40 + n + (50 if official == "Test" else 0))
                    mask = np.zeros((60, 80), np.uint8)
                    mask[10:40, 20:60] = 255
                    target = root / official / "GT_Object" / f"{name}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask).save(target)
                image(root / official / "Image" / "COD10K-NonCAM-ignore.png", 250)
            name = "COD10K-Background-empty"
            image(root / "Train" / "Image" / f"{name}.png", 180)
            Image.fromarray(np.zeros((60, 80), np.uint8)).save(root / "Train" / "GT_Object" / f"{name}.png")
            out = root / "prepared"
            args = dataset.parser().parse_args(["--cod10k", str(root), "--output", str(out)])
            dataset.prepare(args)
            manifest = json.loads((out / "splits.json").read_text())
            self.assertEqual(sum(c["images"] for c in manifest["counts"].values()), 11)
            self.assertEqual(manifest["counts"]["test"]["images"], 2)
            self.assertEqual(len(list((out / "labels").rglob("*Background*.txt"))), 1)
            label = next((out / "labels").rglob("*CAM-Train-0.txt")).read_text().split()
            for value, expected in zip(map(float, label), [0, 0.5, 25 / 60, 0.5, 0.5]):
                self.assertAlmostEqual(value, expected, places=7)
            with self.assertRaises(FileExistsError):
                dataset.prepare(args)

    def test_video_split_keeps_all_frames_but_only_annotated_detection_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            out = prepare_moca(Path(folder))
            manifest = json.loads((out / "splits.json").read_text())
            assignments = {}
            for row in manifest["records"]:
                assignments.setdefault(row["sequence"], set()).add(row["split"])
            self.assertTrue(all(len(v) == 1 for v in assignments.values()))
            self.assertEqual({next(iter(v)) for v in assignments.values()}, {"train", "val", "test"})
            for meta in (out / "tracking").rglob("annotations.json"):
                info = json.loads(meta.read_text())
                self.assertEqual(info["evaluated_frames"], [1, 4])
                self.assertTrue(info["ids_verified"])
                self.assertEqual(len(list((meta.parent / "img1").glob("*.png"))), 4)
                gt = (meta.parent / "gt" / "gt.txt").read_text().splitlines()
                self.assertTrue(all(line.split(",")[1] == "7" for line in gt))
                self.assertEqual([line.split(",")[0] for line in gt], ["1", "4"])
                self.assertIn(",11.0000,11.0000,20.0000,30.0000,", gt[0])
            self.assertEqual(len(list((out / "labels").rglob("*.txt"))), 6)

    def test_missing_ids_are_not_invented_and_evaluation_refuses(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = prepare_moca(root, ids=False)
            for meta in (out / "tracking").rglob("annotations.json"):
                self.assertFalse(json.loads(meta.read_text())["ids_verified"])
                self.assertFalse((meta.parent / "gt" / "gt.txt").exists())
            args = evaluate_camouflage.parser().parse_args([
                str(out / "tracking" / "test"), "--output", str(root / "eval")])
            with self.assertRaisesRegex(ValueError, "추적 ID 정답"):
                evaluate_camouflage.evaluate(args)
            self.assertFalse((root / "eval").exists())

    def test_single_object_confirmation_and_sparse_annotation_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = prepare_moca(root, ids=False, single=True)
            processed = []

            class FakeExtension:
                tracker_config = default_tracker()

                def process(self, frame, number):
                    processed.append(number)
                    # 미주석 프레임의 오검출을 정답 없음으로 잘못 평가하면 점수가 떨어집니다.
                    box = (10, 10, 30, 40) if number in (1, 4) else (40, 20, 60, 40)
                    return [TrackRow(1, CLASS_NAME, box, 0.9, "추적 중", source="camouflage")]

                def close(self):
                    pass

            args = evaluate_camouflage.parser().parse_args([
                str(out / "tracking" / "test"), "--output", str(root / "eval")])
            with mock.patch("evaluate_camouflage.make_extension", return_value=FakeExtension()):
                result = evaluate_camouflage.evaluate(args)
            self.assertEqual(processed, [1, 2, 3, 4])
            self.assertEqual(result["metrics"]["HOTA"], 100)
            self.assertEqual(result["metrics"]["IDF1"], 100)
            self.assertEqual(result["metrics"]["IDSW"], 0)

    def test_bad_csv_ids_boxes_and_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = moca_fixture(root)
            original = csv_path.read_text()
            csv_path.write_text(original.replace("fox,", "../fox,"))
            with self.assertRaises(ValueError):
                dataset.read_moca_csv(csv_path)
            csv_path.write_text(original)
            with self.assertRaises(ValueError):
                dataset.read_moca_csv(csv_path, single_object=True)
            with self.assertRaises(ValueError):
                dataset.clipped_box((0, 0, float("nan"), 20), 80, 60)

    def test_via_derived_csv_requires_explicit_single_object(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "via.csv"
            path.write_text('file_list,spatial_coordinates,metadata\n/crab/00000.jpg,"[2,10,20,30,40]",{}\n')
            self.assertIsNone(dataset.read_moca_csv(path)["crab"]["00000.jpg"][0][1])
            self.assertEqual(dataset.read_moca_csv(path, True)["crab"]["00000.jpg"][0][1], 1)

    def test_cross_split_duplicate_images_abort_without_partial_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = moca_fixture(root)
            for seq in ("crab", "moth"):
                (root / "frames" / seq / "00000.png").write_bytes(
                    (root / "frames" / "fox" / "00000.png").read_bytes())
            args = dataset.parser().parse_args([
                "--moca-frames", str(root / "frames"), "--moca-csv", str(csv_path),
                "--output", str(root / "out")])
            with self.assertRaisesRegex(ValueError, "중복"):
                dataset.prepare(args)
            self.assertFalse((root / "out").exists())


class ExtensionTest(unittest.TestCase):
    def test_existing_pipeline_combines_main_and_camouflage_without_id_collision(self):
        from pipeline import PipelineConfig, TrackingPipeline

        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "camouflage.pt"
            weights.write_bytes(b"fake")
            weights.with_suffix(".json").write_text(json.dumps(
                dict(extension="camouflage", classes=[CLASS_NAME], tracker=default_tracker().to_dict())))

            class FakeDetector:
                def __init__(self, weights, classes, *args):
                    self.names = {0: classes[0]}

                def __call__(self, frame, conf):
                    return np.array([[10, 10, 30, 40.0]]), np.array([0.9]), np.array([0])

            config = PipelineConfig(weights="fake.pt", mosaic=False, extensions=["camouflage"])
            with mock.patch("pipeline.Detector", FakeDetector), \
                    mock.patch.object(CamouflageExtension, "weights_path", return_value=weights):
                pipeline = TrackingPipeline(config)
            try:
                rows, waiting, timing = pipeline.process(np.zeros((60, 80, 3), dtype=np.uint8))
            finally:
                pipeline.close()
            self.assertEqual({(r.source, r.track_id) for r in rows}, {("main", 1), ("camouflage", 1)})
            self.assertEqual({r.cls_name for r in rows}, {"person", CLASS_NAME})
            self.assertIn("extensions", timing)
            self.assertEqual(waiting, [])

    def test_real_tracker_low_score_occlusion_identity_and_result_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weights = root / "camouflage.pt"
            weights.write_bytes(b"fake")
            metadata = dict(extension="camouflage", classes=[CLASS_NAME], tracker=default_tracker().to_dict())
            weights.with_suffix(".json").write_text(json.dumps(metadata))

            class FakeDetector:
                names = {0: CLASS_NAME}

                def __init__(self, *args):
                    self.number = 0

                def __call__(self, frame, conf):
                    self.number += 1
                    if self.number in (4, 5):
                        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
                    x = 10 + self.number * 2
                    return np.array([[x, 10, x + 20, 40.0]]), np.array([0.15 if self.number == 3 else 0.9]), np.array([0])

            with mock.patch("pipeline.Detector", FakeDetector):
                extension = run_camouflage.make_extension(weights)
            frame = np.full((60, 80, 3), 100, dtype=np.uint8)
            original = frame.copy()
            rows = [extension.process(frame, n) for n in range(1, 9)]
            self.assertTrue(rows[2])  # 낮은 점수 검출을 2차 연결로 유지
            self.assertFalse(rows[3])  # 완전 가림을 실제 검출로 기록하지 않음
            self.assertEqual({r.track_id for rs in rows for r in rs}, {1})
            self.assertTrue(all(r.source == "camouflage" for rs in rows for r in rs))
            self.assertGreater(rows[-1][0].distance, 0)
            self.assertEqual(rows[-1][0].direction, "→")
            np.testing.assert_array_equal(frame, original)
            from results_io import ResultWriter

            writer = ResultWriter()
            for n, rs in enumerate(rows, 1):
                writer.add(n, n * 1000 / 30, rs)
            writer.save(root / "tracks.csv", {"source": "test.mp4"})
            replay, _ = load_results(root / "tracks.csv")
            self.assertEqual(replay[8][0]["label"], "보호색 동물")
            self.assertEqual(replay[8][0]["source"], "camouflage")

    def test_missing_or_invalid_metadata_is_explained_without_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camouflage.pt"

            class Selected(CamouflageExtension):
                @classmethod
                def weights_path(cls):
                    return path

            self.assertIn("가중치가 없습니다", Selected.check())
            path.write_bytes(b"fake")
            self.assertIn("JSON", Selected.check())
            path.with_suffix(".json").write_text(json.dumps(dict(extension="camouflage", classes=["person"])))
            self.assertIsNotNone(Selected.check())
            data = dict(extension="camouflage", classes=[CLASS_NAME], tracker={"high_thresh": 0.01})
            path.with_suffix(".json").write_text(json.dumps(data))
            self.assertIsNotNone(Selected.check())


class TrainingTest(unittest.TestCase):
    def test_train_validates_test_split_and_installs_matching_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = prepare_moca(root)
            best = root / "best.pt"
            best.write_bytes(b"fake")
            calls = []

            class FakeYOLO:
                names = {0: CLASS_NAME}
                trainer = SimpleNamespace(best=best)

                def __init__(self, path):
                    pass

                def train(self, **options):
                    calls.append(options)

                def val(self, **options):
                    calls.append(options["split"])
                    return SimpleNamespace(box=SimpleNamespace(map50=0.5, map=0.3))

            args = train_camouflage.parser().parse_args(["--data", str(out / "dataset.yaml"), "--install"])
            with mock.patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=FakeYOLO)}), \
                    mock.patch("train_camouflage.ROOT", root):
                train_camouflage.train(args)
            metadata = json.loads((root / "models" / "camouflage.json").read_text())
            self.assertEqual(calls[1:], ["val", "test"])
            self.assertEqual(metadata["classes"], [CLASS_NAME])
            self.assertEqual(metadata["scores"]["test"]["mAP50"], 0.5)
            self.assertEqual(len(metadata["weights_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                train_camouflage.install_model(best, metadata, root / "models")


class VideoTest(unittest.TestCase):
    def test_video_outputs_can_be_replayed_without_a_model(self):
        import cv2

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
            self.assertTrue(writer.isOpened())
            for n in range(4):
                writer.write(np.full((60, 80, 3), n * 30, dtype=np.uint8))
            writer.release()

            class FakeExtension:
                tracker_config = default_tracker()

                def process(self, frame, number):
                    return [TrackRow(2, CLASS_NAME, (10, 10, 30, 40), 0.8, "추적 중",
                                     trail=((10, 10), (20, 20)), source="camouflage", label="보호색 동물")]

                def close(self):
                    pass

            args = run_camouflage.parser().parse_args([
                "--video", str(source), "--output", str(root / "result")])
            with mock.patch("run_camouflage.make_extension", return_value=FakeExtension()):
                meta = run_camouflage.run(args)
            rows, saved = load_results(root / "result" / "tracks.csv")
            self.assertEqual(meta["frames"], 4)
            self.assertEqual(saved["fps"], 10)
            self.assertEqual(set(rows), {1, 2, 3, 4})
            self.assertEqual(rows[4][0]["source"], "camouflage")
            cap = cv2.VideoCapture(str(root / "result" / "tracked.mp4"))
            try:
                self.assertTrue(cap.isOpened())
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            finally:
                cap.release()
            with self.assertRaises(FileExistsError):
                run_camouflage.run(args)


if __name__ == "__main__":
    unittest.main()
