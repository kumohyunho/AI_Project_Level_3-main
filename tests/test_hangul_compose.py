"""자모 → 글자·단어 조합과 시연 영상 생성기 테스트.  실행: ``python -m unittest discover -s tests -t .``"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hangul_dataset  # noqa: E402
from extensions.hangul import CLASS_NAMES, JamoVotes  # noqa: E402
from extensions.hangul_compose import (Jamo, WordStabilizer, compose, decompose, group_syllables,  # noqa: E402
                                       read_words)
from hangul_demo import syllable_layout, word_layout  # noqa: E402
from tests.test_hangul_extension import make_extension  # noqa: E402

# 자모 그림의 대략적인 크기 (자음 폭 g 기준): 세로 모음은 가늘고 길며, 가로 모음은 넓고 낮습니다.
SHAPE = {"ㅏ": (0.6, 1.4), "ㅐ": (0.65, 1.4), "ㅓ": (0.6, 1.4), "ㅔ": (0.75, 1.4), "ㅑ": (0.6, 1.4),
         "ㅣ": (0.3, 1.4), "ㅡ": (1.3, 0.3), "ㅜ": (1.3, 0.8), "ㅗ": (1.3, 0.8)}


def boxes_for(word, g=60.0, shift=(0.0, 0.0)):
    """hangul_demo의 배치대로 놓은 자모 박스 목록."""
    items = []
    for index, (char, cx, cy) in enumerate(word_layout(word, 960, 540, g), start=1):
        w, h = SHAPE.get(char, (1.0, 0.85))
        cx, cy = cx + shift[0], cy + shift[1]
        items.append(Jamo(index, char, (cx - w * g / 2, cy - h * g / 2, cx + w * g / 2, cy + h * g / 2)))
    return items


class ComposeTest(unittest.TestCase):
    def test_compose_and_decompose(self):
        self.assertEqual(compose("ㅎ", "ㅏ", "ㄴ"), "한")
        self.assertEqual(compose("ㄱ", "ㅡ", "ㄹ"), "글")
        self.assertEqual(compose("ㅊ", "ㅔ"), "체")
        self.assertEqual(compose("ㄲ", "ㅘ"), "꽈")
        for syllable in "한글객체추적인공지능꽈뷁":
            self.assertEqual(compose(*decompose(syllable)), syllable)
        with self.assertRaises(ValueError):
            compose("ㅏ", "ㄱ")
        with self.assertRaises(ValueError):
            compose("ㄱ", "ㅏ", "ㄸ")   # ㄸ은 종성이 될 수 없음

    def test_demo_layout_is_read_back(self):
        # 시연 영상의 배치 규칙과 조합 규칙이 서로 맞아야 합니다.
        for word in ("한글", "객체추적", "인공지능", "사과", "꿈"):
            words = read_words(boxes_for(word))
            self.assertEqual([w.text for w in words], [word], word)

    def test_two_words_on_separate_lines(self):
        items = boxes_for("한글") + [Jamo(j.id + 100, j.char, j.box) for j in boxes_for("객체", shift=(0, 220))]
        self.assertEqual([w.text for w in read_words(items)], ["한글", "객체"])

    def test_scattered_jamo_are_not_composed(self):
        rng = np.random.default_rng(0)
        items = []
        for index, char in enumerate("ㅎㅏㄴㄱㅡㄹ"):
            x, y = rng.uniform(0, 900), rng.uniform(0, 500)
            items.append(Jamo(index, char, (x, y, x + 60, y + 60)))
        # 서로 멀리 떨어진 자모는 글자가 되지 않습니다 (가장 가까운 두 자모도 자모 크기의 몇 배 떨어짐).
        far = [Jamo(i, c, (i * 200.0, 0.0, i * 200.0 + 60, 60.0)) for i, c in enumerate("ㅎㅏㄴㄱㅡㄹ")]
        self.assertEqual(read_words(far), [])
        syllables, leftover = group_syllables(items)
        self.assertEqual(len(leftover) + sum(len(s.members) for s in syllables), 6)

    def test_consonant_on_wrong_side_is_not_initial(self):
        # ㄱ이 ㅏ의 오른쪽에 있으면 초성이 아닙니다 ("ㅏㄱ"은 글자가 아님).
        items = [Jamo(1, "ㅏ", (100, 100, 136, 184)), Jamo(2, "ㄱ", (150, 110, 210, 160))]
        self.assertEqual(read_words(items), [])


class StabilizerTest(unittest.TestCase):
    def test_needs_min_frames_and_keeps_id(self):
        stabilizer = WordStabilizer(min_frames=3)
        word = read_words(boxes_for("한글"))
        self.assertEqual(stabilizer.update(word), [])
        self.assertEqual(stabilizer.update(word), [])
        shown = stabilizer.update(word)
        self.assertEqual([(i, w.text) for i, w in shown], [(1001, "한글")])
        self.assertEqual([i for i, _ in stabilizer.update(word)], [1001])

    def test_brief_misreading_does_not_flicker(self):
        stabilizer = WordStabilizer(min_frames=3)
        good, bad = read_words(boxes_for("한글")), read_words(boxes_for("햔글"))
        for _ in range(5):
            stabilizer.update(good)
        for _ in range(2):   # ㅏ가 두 프레임 ㅑ로 읽힘
            shown = stabilizer.update(bad)
            self.assertEqual([w.text for _, w in shown], ["한글"])

    def test_disappears_after_missing_frames(self):
        stabilizer = WordStabilizer(min_frames=2, max_missing=2)
        word = read_words(boxes_for("한글"))
        stabilizer.update(word)
        self.assertTrue(stabilizer.update(word))
        for _ in range(3):
            shown = stabilizer.update([])
        self.assertEqual(shown, [])


class VoteDecayTest(unittest.TestCase):
    def test_decay_lets_recent_jamo_win(self):
        full, decayed = JamoVotes(1.0), JamoVotes(0.9)
        for votes in (full, decayed):
            for _ in range(20):
                votes.add(1, "giyeok", 0.9)   # ID가 뒤바뀌기 전: ㄱ
            for _ in range(15):
                votes.add(1, "hieut", 0.9)    # 뒤바뀐 뒤: 실제로는 ㅎ
        self.assertEqual(full.best(1), "giyeok")   # 전체 다수결은 예전 자모가 남음
        self.assertEqual(decayed.best(1), "hieut")  # 감쇠하면 최근 자모로 바뀜


class ExtensionWordTest(unittest.TestCase):
    def run_scene(self, moving):
        """'한' 모양으로 놓인 자모 세 개. ``moving``이면 함께 오른쪽으로 계속 움직입니다."""
        layout = boxes_for("한")
        ids = {"ㅎ": CLASS_NAMES.index("hieut"), "ㅏ": CLASS_NAMES.index("a"), "ㄴ": CLASS_NAMES.index("nieun")}

        def script(index):
            dx = 6.0 * index if moving else 0.0
            boxes = [[j.box[0] + dx, j.box[1], j.box[2] + dx, j.box[3]] for j in layout]
            return boxes, [0.9] * len(layout), [ids[j.char] for j in layout]

        extension = make_extension(script)
        rows = []
        for index in range(1, 25):
            boxes, _, _ = script(index)
            frame = np.full((540, 960, 3), 90, dtype=np.uint8)
            for k, (x1, y1, x2, y2) in enumerate(boxes):
                frame[int(y1):int(y2), int(x1):int(x2)] = (40 + 60 * k, 200 - 50 * k, 120)
            rows = extension.process(frame, index)
        return rows

    def test_stationary_jamo_become_a_word(self):
        rows = self.run_scene(moving=False)
        words = [r for r in rows if r.cls_name == "단어"]
        self.assertEqual([(r.label, r.source) for r in words], [("한", "hangul")])
        self.assertGreaterEqual(words[0].track_id, 1001)
        self.assertEqual(sorted(r.label for r in rows if r.cls_name != "단어"), sorted("ㅎㅏㄴ"))

    def test_moving_jamo_are_not_composed(self):
        rows = self.run_scene(moving=True)
        self.assertEqual([r for r in rows if r.cls_name == "단어"], [])


@unittest.skipUnless(hangul_dataset.find_fonts(), "한글 글꼴이 없습니다 (Linux: apt-get install fonts-nanum)")
class DemoVideoTest(unittest.TestCase):
    def test_short_demo_video(self):
        import cv2

        from hangul_demo import make_demo

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "demo.mp4"
            info = make_demo(["한글"], output, float_frames=5, gather_frames=5, hold_frames=5)
            self.assertEqual(info["frames"], 20)
            self.assertEqual([p["kind"] for p in info["phases"]], ["떠다님", "모임", "멈춤", "흩어짐"])
            saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(saved["phases"][2]["word"], "한글")
            capture = cv2.VideoCapture(str(output))
            count = 0
            while capture.read()[0]:
                count += 1
            self.assertEqual(count, 20)

    def test_unsupported_syllable_is_rejected(self):
        with self.assertRaises(ValueError):
            syllable_layout("닭", 0, 0, 60)   # 겹받침 ㄺ은 검출 대상이 아님


if __name__ == "__main__":
    unittest.main()
