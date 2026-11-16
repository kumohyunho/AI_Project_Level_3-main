"""추적기·지표·결과 파일 단위 테스트.  실행: ``python -m unittest discover tests -v``"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import combine, evaluate_files  # noqa: E402
from results_io import write_mot  # noqa: E402
from tests.synthetic import make_sequence, write_gt  # noqa: E402
from tracker import ByteTracker, MotionRecorder, TrackerConfig, extract_features  # noqa: E402
from tracker.kalman import KalmanFilter  # noqa: E402


def run(config, images, detections, use_features):
    tracker = ByteTracker(config, 30)
    rows = []
    for number, image in enumerate(images, start=1):
        boxes, scores = detections[number]
        features = extract_features(image, boxes) if use_features else None
        for t in tracker.update(boxes, scores, np.zeros(len(boxes)), features, (image.shape[1], image.shape[0])):
            rows.append((number, t.track_id, *t.box, t.score))
    return rows


def evaluate(rows, gt, folder, name):
    gt_path, result_path = Path(folder) / "gt.txt", Path(folder) / f"{name}.txt"
    write_gt(gt_path, gt)
    write_mot(result_path, rows)
    return combine([evaluate_files(gt_path, result_path, sorted(gt))])


class KalmanTest(unittest.TestCase):
    def test_constant_velocity_prediction(self):
        kalman = KalmanFilter()
        mean, cov = kalman.initiate(np.array([100.0, 100.0, 0.5, 80.0]))
        for step in range(1, 15):
            mean, cov = kalman.predict(mean, cov)
            mean, cov = kalman.update(mean, cov, np.array([100.0 + 5 * step, 100.0, 0.5, 80.0]))
        predicted, _ = kalman.predict(mean, cov)
        self.assertAlmostEqual(predicted[4], 5.0, delta=0.5)   # 속도 추정
        self.assertAlmostEqual(predicted[0], 175.0, delta=2.0)  # 다음 위치 예측


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.gt = {f: [(1, np.array([10, 10, 50, 90.0]) + f, 1.0),
                       (2, np.array([200, 10, 240, 90.0]) - f, 1.0)] for f in range(1, 31)}

    def test_perfect_tracking(self):
        rows = [(f, pid, *box, 1.0) for f, items in self.gt.items() for pid, box, _ in items]
        result = evaluate(rows, self.gt, self.folder, "perfect")
        for key in ("HOTA", "MOTA", "IDF1"):
            self.assertAlmostEqual(result[key], 100.0, places=4)
        self.assertEqual(result["IDSW"], 0)

    def test_id_switch_counted(self):
        rows = []
        for f, items in self.gt.items():
            for pid, box, _ in items:
                track_id = pid if f <= 15 else 3 - pid   # 16프레임부터 두 ID가 뒤바뀜
                rows.append((f, track_id, *box, 1.0))
        result = evaluate(rows, self.gt, self.folder, "swap")
        self.assertEqual(result["IDSW"], 2)
        self.assertAlmostEqual(result["MOTA"], 100 * (1 - 2 / 60), places=4)
        self.assertAlmostEqual(result["IDF1"], 50.0, places=4)
        self.assertLess(result["HOTA"], 80)

    def test_missed_frames_are_false_negatives(self):
        rows = [(f, pid, *box, 1.0) for f, items in self.gt.items() for pid, box, _ in items if f % 2]
        result = evaluate(rows, self.gt, self.folder, "half")
        self.assertEqual(result["FN"], 30)
        self.assertAlmostEqual(result["MOTA"], 50.0, places=4)


class TrackerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.images, cls.gt, cls.detections = make_sequence()
        cls.folder = tempfile.mkdtemp()
        cls.baseline = evaluate(run(TrackerConfig.preset("baseline"), cls.images, cls.detections, False),
                                cls.gt, cls.folder, "baseline")
        cls.improved = evaluate(run(TrackerConfig.preset("improved"), cls.images, cls.detections, True),
                                cls.gt, cls.folder, "improved")
        print(f"\n[합성 장면] baseline {cls.baseline}\n[합성 장면] improved {cls.improved}")

    def test_baseline_tracks_people(self):
        self.assertGreater(self.baseline["MOTA"], 50)

    def test_improved_keeps_identity_better(self):
        self.assertGreaterEqual(self.improved["IDF1"], self.baseline["IDF1"])
        self.assertLessEqual(self.improved["IDs"], self.baseline["IDs"])
        self.assertGreaterEqual(self.improved["HOTA"], self.baseline["HOTA"])

    def test_class_gate_keeps_classes_apart(self):
        tracker = ByteTracker(TrackerConfig(), 30)
        box = np.array([[10, 10, 50, 90.0]])
        for _ in range(3):
            tracks = tracker.update(box, np.array([0.9]), np.array([0]))
        person_id = tracks[0].track_id
        tracks = tracker.update(box, np.array([0.9]), np.array([16]))  # 같은 자리, 다른 클래스
        self.assertNotIn(person_id, [t.track_id for t in tracks])

    def test_lost_track_status(self):
        tracker = ByteTracker(TrackerConfig(), 30)
        for x in range(0, 30, 3):  # 오른쪽 끝으로 이동하다 사라짐
            tracker.update(np.array([[580 + x, 100, 620 + x, 180.0]]), np.array([0.9]), np.array([0]),
                           frame_size=(640, 360))
        tracker.update(np.zeros((0, 4)), np.zeros(0), np.zeros(0), frame_size=(640, 360))
        self.assertEqual(tracker.lost_tracks()[0].status, "화면 이탈")


class MotionTest(unittest.TestCase):
    def test_direction_distance_speed(self):
        motion = MotionRecorder(fps=10)
        for frame in range(1, 6):
            state = motion.update(7, (10 * frame, 0, 10 * frame + 10, 10), frame)
        self.assertEqual(state.direction, "→")
        self.assertAlmostEqual(state.distance, 40.0)
        self.assertGreater(state.speed, 50)
        self.assertEqual(len(state.trail), 5)


if __name__ == "__main__":
    unittest.main()
