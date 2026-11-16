"""한글 자모 확장 기능 테스트: 클래스가 프레임마다 바뀌어도 ID가 유지되고 자모는 다수결로 정해지는지.

실행: ``python -m unittest discover -s tests -t .``
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extensions.hangul import CLASS_NAMES, HangulJamoExtension, JamoVotes  # noqa: E402

A, EO, GIYEOK = CLASS_NAMES.index("a"), CLASS_NAMES.index("eo"), CLASS_NAMES.index("giyeok")


class FakeDetector:
    def __init__(self, weights, classes, imgsz=640, device="cpu"):
        self.names = dict(enumerate(classes))
        self.class_ids = list(range(len(classes)))


def make_extension(script):
    """``script(frame_index) -> (boxes, scores, class_ids)``로 검출을 대신하는 확장 기능."""
    weights = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    weights.close()

    class Scripted(HangulJamoExtension):
        @classmethod
        def weights_path(cls):
            return Path(weights.name)

        def detect(self, frame):
            self.calls = getattr(self, "calls", 0) + 1
            boxes, scores, classes = script(self.calls)
            return np.array(boxes, dtype=float).reshape(-1, 4), np.array(scores, dtype=float), np.array(classes, dtype=int)

    with mock.patch("pipeline.Detector", FakeDetector):
        extension = Scripted(fps=30)
    Path(weights.name).unlink()
    return extension


def colored_frame(boxes, colors):
    frame = np.full((240, 480, 3), 60, dtype=np.uint8)
    for (x1, y1, x2, y2), color in zip(boxes, colors):
        frame[int(y1):int(y2), int(x1):int(x2)] = color
    return frame


class VotesTest(unittest.TestCase):
    def test_majority_weighted_by_score(self):
        votes = JamoVotes()
        for name, score in [("a", 0.9), ("eo", 0.6), ("a", 0.8), ("eo", 0.7), ("a", 0.9)]:
            votes.add(1, name, score)
        self.assertEqual(votes.best(1), "a")
        self.assertAlmostEqual(votes.confidence(1), 2.6 / 3.9)
        self.assertEqual(votes.frames[1], 5)
        self.assertEqual(votes.disagree[1], 2)
        self.assertIsNone(votes.best(2))


class JamoTrackingTest(unittest.TestCase):
    def test_class_flicker_keeps_id_and_majority_label(self):
        # ㅏ 하나가 오른쪽으로 움직입니다. 5프레임마다 ㅓ로 잘못 검출됩니다 (회전 때문에 생기는 상황).
        def script(index):
            x = 20 + 6 * index
            return [[x, 80, x + 40, 140]], [0.9], [EO if index % 5 == 0 else A]

        extension = make_extension(script)
        seen = []
        for index in range(1, 31):
            boxes, _, _ = script(index)
            seen += extension.process(colored_frame(boxes, [(40, 40, 220)]), index)
        self.assertEqual({row.track_id for row in seen}, {1})           # 클래스가 바뀌어도 ID 하나
        self.assertTrue(all(row.label == "ㅏ" for row in seen[1:]))     # 표시 자모는 다수결
        self.assertTrue(all(row.cls_name == "모음" and row.source == "hangul" for row in seen))
        self.assertGreater(extension.votes.disagree[1], 0)
        self.assertEqual(seen[-1].direction, "→")

    def test_two_objects_get_separate_ids_and_labels(self):
        def script(index):
            return ([[20 + 5 * index, 30, 60 + 5 * index, 90], [400 - 5 * index, 150, 440 - 5 * index, 210]],
                    [0.9, 0.85], [GIYEOK, A])

        extension = make_extension(script)
        rows = []
        for index in range(1, 21):
            boxes, _, _ = script(index)
            rows = extension.process(colored_frame(boxes, [(30, 200, 230), (200, 60, 200)]), index)
        self.assertEqual(sorted((row.track_id, row.label, row.cls_name) for row in rows),
                         [(1, "ㄱ", "자음"), (2, "ㅏ", "모음")])

    def test_low_score_detections_do_not_vote(self):
        # 가려져 점수가 낮아진 프레임의 엉뚱한 분류는 투표에 넣지 않습니다.
        def script(index):
            x = 20 + 4 * index
            low = 10 <= index <= 14
            return [[x, 80, x + 40, 140]], [0.3 if low else 0.9], [EO if low else A]

        extension = make_extension(script)
        for index in range(1, 21):
            boxes, _, _ = script(index)
            rows = extension.process(colored_frame(boxes, [(40, 40, 220)]), index)
        self.assertEqual([(row.track_id, row.label) for row in rows], [(1, "ㅏ")])
        self.assertEqual(extension.votes.disagree[1], 0)

    def test_no_detections(self):
        extension = make_extension(lambda index: ([], [], []))
        self.assertEqual(extension.process(np.zeros((100, 100, 3), dtype=np.uint8), 1), [])


class AvailabilityTest(unittest.TestCase):
    def test_disabled_only_when_weights_missing(self):
        reason = HangulJamoExtension.check()
        if HangulJamoExtension.weights_path().is_file():
            self.assertIsNone(reason)
        else:
            self.assertIn("models/hangul_jamo.pt", reason)


if __name__ == "__main__":
    unittest.main()
