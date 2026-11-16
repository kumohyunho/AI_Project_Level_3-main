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

    def reentry_scenario(self, returning_class, verify="reentry"):
        """ㅏ가 오른쪽 밖으로 나간 뒤, 같은 색의 객체가 같은 자리로 다시 들어옵니다 (추적기는 이전 ID를 돌려줌)."""
        def script(index):
            if index <= 20:                      # 오른쪽으로 나가는 ㅏ
                x = 300 + 9 * index
            elif index <= 70:                    # 화면 밖 (Lost → 삭제되어 외형이 보관됨)
                return [], [], []
            else:                                # 다시 들어오는 객체
                x = 480 - 9 * (index - 70)
            x1, x2 = max(0, x), min(480, x + 40)
            if x2 - x1 < 16:                     # 거의 다 나간 객체는 검출되지 않습니다.
                return [], [], []
            return [[x1, 80, x2, 140]], [0.9], [A if index <= 20 else returning_class]

        extension = make_extension(script)
        extension.verify = verify
        history = []
        for index in range(1, 91):
            boxes, _, _ = script(index)
            history.append(extension.process(colored_frame(boxes, [(40, 40, 220)]), index))
        before = {(r.track_id, r.label) for rows in history[:20] for r in rows}
        after = {(r.track_id, r.label) for rows in history[84:] for r in rows}   # 판단(검출 8번)이 끝난 뒤
        return extension, before, after

    def test_restored_id_kept_for_same_jamo(self):
        extension, before, after = self.reentry_scenario(A)
        self.assertEqual(before, {(1, "ㅏ")})
        if extension.byte_tracker.next_id > 2:
            self.skipTest("이 장면에서 추적기가 이전 ID를 복원하지 않았습니다.")
        self.assertEqual(after, {(1, "ㅏ")})
        self.assertEqual(extension.rejected, 0)

    def test_restored_id_rejected_for_different_jamo(self):
        extension, before, after = self.reentry_scenario(GIYEOK)
        self.assertEqual(before, {(1, "ㅏ")})
        self.assertEqual(after, {(2, "ㄱ")})      # 이전 ID 1과 ㅏ가 따라오지 않아야 합니다.
        if extension.byte_tracker.next_id == 2:  # 추적기는 이전 ID를 돌려줬고, 확장 기능이 취소한 경우
            self.assertEqual(extension.rejected, 1)

    def test_without_check_votes_follow_the_restored_id(self):
        extension, before, after = self.reentry_scenario(GIYEOK, verify="off")
        if extension.byte_tracker.next_id > 2:
            self.skipTest("이 장면에서 추적기가 이전 ID를 복원하지 않았습니다.")
        self.assertIn((1, "ㅏ"), after)           # 확인을 끄면 틀린 자모가 표시됩니다 (고치려는 문제).

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
