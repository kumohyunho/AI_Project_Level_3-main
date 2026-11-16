"""한글 자모 평가 스크립트 테스트: 정답 박스를 검출로 써서(모델 없이) 지표 계산을 확인합니다.

실행: ``python -m unittest discover -s tests -t .``
"""
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_hangul  # noqa: E402
import hangul_dataset  # noqa: E402
from extensions.hangul import CLASS_NAMES  # noqa: E402
from metrics import evaluate_files  # noqa: E402
from mot_data import Sequence  # noqa: E402
from results_io import write_mot  # noqa: E402
from tests.test_hangul_extension import make_extension  # noqa: E402

CHAR_TO_ID = {char: index for index, char in enumerate(evaluate_hangul.CHAR.values())}


def oracle_detections(sequence, truth, wrong_every=0):
    """정답 박스를 검출로 씁니다. ``wrong_every``번째 프레임마다 모든 자모를 다음 클래스로 잘못 분류합니다."""
    frames, chars, _ = truth
    detections = {}
    for number in sequence.frames:
        items = [item for item in frames.get(number, []) if item[2] >= 0.3]
        wrong = wrong_every and number % wrong_every == 0
        classes = [(CHAR_TO_ID[chars[i]] + (1 if wrong else 0)) % len(CLASS_NAMES) for i, _, _ in items]
        detections[number] = (np.array([box for _, box, _ in items], dtype=float).reshape(-1, 4),
                              np.full(len(items), 0.9), np.array(classes, dtype=int), [None] * len(items))
    return detections


@unittest.skipUnless(hangul_dataset.find_fonts(), "한글 글꼴이 없습니다 (Linux: apt-get install fonts-nanum)")
class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        output = Path(cls.folder.name) / "hangul"
        hangul_dataset.main(Namespace(output=str(output), train=1, val=1, test=1, sequences=1, length=60,
                                      backgrounds=None, fonts=None, seed=3, rotation=45.0))
        cls.sequence = Sequence(output / "tracking" / "test" / "jamo-01")
        cls.truth = evaluate_hangul.load_truth(cls.sequence)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def run_oracle(self, wrong_every):
        extension = make_extension(lambda index: ([], [], []))
        detections = oracle_detections(self.sequence, self.truth, wrong_every)
        rows, shown, final, _ = evaluate_hangul.run_tracker(extension, self.sequence, detections, "baseline")
        records = evaluate_hangul.classify(self.sequence, self.truth, shown, final)
        return rows, evaluate_hangul.summarize(records)

    def test_truth_is_zero_based_and_labelled(self):
        frames, chars, angles = self.truth
        self.assertTrue(all(box[0] >= 0 and box[1] >= 0 for items in frames.values() for _, box, _ in items))
        self.assertTrue(set(chars.values()) <= set(CHAR_TO_ID))
        self.assertTrue(all(abs(angle) <= 45 for angle in angles.values()))

    def test_perfect_detections_give_perfect_classification(self):
        rows, (summary, per_jamo, confusion_frame, confusion_shown) = self.run_oracle(wrong_every=0)
        self.assertEqual(summary["frame_accuracy"], 100.0)
        self.assertEqual(summary["shown_accuracy"], 100.0)
        self.assertEqual(summary["track_accuracy"], 100.0)
        self.assertGreater(summary["detection_rate"], 85.0)
        self.assertFalse(confusion_frame or confusion_shown)
        self.assertEqual(sum(v["gt_boxes"] for v in summary["by_angle"].values()), summary["gt_boxes"])
        self.assertEqual(sum(j["gt_boxes"] for j in per_jamo), summary["gt_boxes"])
        # 결과 좌표(0부터)와 정답 파일(1부터)의 기준이 맞아야 추적 지표의 위치 정확도가 100에 가깝습니다.
        target = Path(self.folder.name) / "oracle.txt"
        write_mot(target, rows)
        metrics = evaluate_files(self.sequence.gt_path, target, self.sequence.frames)
        self.assertGreater(metrics["LocA"] if "LocA" in metrics else 100.0, 95.0)

    def test_majority_vote_hides_occasional_wrong_frames(self):
        _, (summary, _, confusion_frame, _) = self.run_oracle(wrong_every=4)
        self.assertLess(summary["frame_accuracy"], 80.0)                  # 4프레임마다 틀린 분류
        self.assertGreater(summary["shown_accuracy_voting_frames"], 95.0)  # 다수결은 거의 영향 없음
        self.assertGreater(summary["shown_accuracy_voting_frames"], summary["frame_accuracy"])
        self.assertTrue(confusion_frame)

    def test_same_scene_for_every_rotation_range(self):
        # --tracking-only: 검출 이미지는 만들지 않고, seed가 같으면 회전 범위만 다른 같은 장면이 나와야 합니다.
        objects, angles = {}, {}
        for rotation in (0.0, 90.0):
            output = Path(self.folder.name) / f"rot{rotation:g}"
            hangul_dataset.main(Namespace(output=str(output), train=5, val=5, test=5, sequences=1, length=30,
                                          backgrounds=None, fonts=None, seed=3, rotation=rotation, tracking_only=True))
            self.assertEqual(list((output / "images" / "train").glob("*.jpg")), [])
            folder = output / "tracking" / "test" / "jamo-01" / "gt"
            objects[rotation] = (folder / "objects.json").read_text(encoding="utf-8")
            rows = (folder / "angles.csv").read_text().splitlines()[1:]
            angles[rotation] = max(abs(float(row.split(",")[2])) for row in rows)
        self.assertEqual(objects[0.0], objects[90.0])
        self.assertEqual(angles[0.0], 0.0)
        self.assertGreater(angles[90.0], 45.0)

    def test_failure_events_split_crossings_by_jamo(self):
        frames, chars, _ = self.truth
        # 실제 정답에서 서로 겹치는 두 객체를 찾아, 그 프레임에 ID 변경이 났다고 가정합니다.
        from tracker.matching import iou_matrix
        found = None
        for number, items in frames.items():
            boxes = [box for _, box, _ in items]
            overlap = iou_matrix(boxes, boxes)
            np.fill_diagonal(overlap, 0)
            if overlap.max() >= 0.3:
                a, b = np.unravel_index(overlap.argmax(), overlap.shape)
                found = (number, items[a][0], items[b][0])
                break
        if found is None:
            self.skipTest("이 장면에는 겹치는 자모가 없습니다.")
        number, first, second = found
        events = [dict(type="IDSW", frame=number, gt_id=first, old_id=1, new_id=2)]
        evaluate_hangul.classify_events(events, self.sequence, chars)
        expected = "교차(같은 자모)" if chars[first] == chars[second] else "교차(다른 자모)"
        self.assertEqual(events[0]["category"], expected)
        self.assertEqual(events[0]["char"], chars[first])
        self.assertIn(events[0]["category"], evaluate_hangul.FAILURE_CATEGORIES)

    def test_wrong_tracks_and_run_comparison(self):
        record = dict(sequence="s", matched=True, track_id=1, char="ㅏ", shown="ㅓ", observed="ㅓ", final="ㅓ", angle=30.0)
        good = dict(record, track_id=2, shown="ㅏ", observed="ㅏ", final="ㅏ")
        wrong = evaluate_hangul.wrong_tracks([record, record, good])
        self.assertEqual([(w["track_id"], w["char"], w["final"], w["frames"]) for w in wrong], [(1, "ㅏ", "ㅓ", 2)])

        _, (summary, per_jamo, _, _) = self.run_oracle(wrong_every=0)
        root = Path(self.folder.name) / "runs"
        tracking = dict(HOTA=50.0, IDF1=60.0, MOTA=70.0, IDSW=1)
        for name in ("rot0", "rot90"):
            (root / name).mkdir(parents=True)
            (root / name / "summary.json").write_text(json.dumps(
                {"results": {"baseline": {"tracking": tracking, "classification": summary}}}), encoding="utf-8")
        text = evaluate_hangul.summarize_runs(["rot0", "rot90"], root)
        self.assertIn("| rot0 | baseline | 50.00 |", text)
        self.assertIn("| rot90 | baseline |", text)
        self.assertTrue((root / "comparison.md").is_file())
        with self.assertRaises(FileNotFoundError):
            evaluate_hangul.summarize_runs(["없는평가"], root)

    def test_report_is_written(self):
        _, (summary, per_jamo, confusion_frame, confusion_shown) = self.run_oracle(wrong_every=4)
        tracking = dict(HOTA=1.0, IDF1=2.0, MOTA=3.0, IDSW=0, Frag=0, GT_IDs=5, IDs=5)
        results = {"baseline": dict(tracking=tracking, classification=summary, per_jamo=per_jamo,
                                    confusion_frame=confusion_frame, confusion_shown=confusion_shown, track_ms=0.1)}
        folder = Path(self.folder.name)
        text = evaluate_hangul.write_report(folder, "unit", "x.pt", [self.sequence], results, 1.0)
        self.assertIn("다수결 효과", text)
        self.assertIn("회전 각도별", text)
        self.assertTrue((folder / "summary.md").is_file())
        json.dumps(summary, ensure_ascii=False)  # JSON으로 저장할 수 있어야 합니다.


if __name__ == "__main__":
    unittest.main()
