"""한글 자모 합성 데이터 생성기 테스트 (한글 글꼴이 없으면 건너뜁니다).  실행: ``python -m unittest discover -s tests -t .``"""
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hangul_dataset  # noqa: E402
from extensions.hangul import CHAR, CLASS_NAMES, JAMO  # noqa: E402
from mot_data import Sequence  # noqa: E402

FONTS = hangul_dataset.find_fonts()


class JamoTableTest(unittest.TestCase):
    def test_forty_distinct_jamo(self):
        self.assertEqual(len(JAMO), 40)
        self.assertEqual(len(set(CLASS_NAMES)), 40)
        self.assertEqual(len(set(CHAR.values())), 40)
        self.assertEqual(sum(kind == "자음" for _, _, kind in JAMO), 19)
        self.assertTrue(all(name.isascii() and 0x3131 <= ord(char) <= 0x3163 for name, char, _ in JAMO))


class FontSplitTest(unittest.TestCase):
    def test_small_font_set_is_shared(self):
        fonts = ["a.ttf", "b.ttf"]
        self.assertEqual(hangul_dataset.split_fonts(fonts)["test"], fonts)

    def test_large_font_set_holds_out_test_fonts(self):
        fonts = [f"font{i}.ttf" for i in range(10)]
        split = hangul_dataset.split_fonts(fonts)
        self.assertTrue(split["test"])
        self.assertFalse(set(split["test"]) & set(split["train"]))
        self.assertEqual(set(split["test"]) | set(split["train"]), set(fonts))


@unittest.skipUnless(FONTS, "한글 글꼴이 없습니다 (Linux: apt-get install fonts-nanum)")
class GeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.output = Path(cls.folder.name) / "hangul"
        hangul_dataset.main(Namespace(output=str(cls.output), train=6, val=2, test=2, sequences=1, length=40,
                                      backgrounds=None, fonts=None, seed=1, rotation=45.0))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_detection_labels_match_annotations(self):
        for split, count in (("train", 6), ("val", 2), ("test", 2)):
            notes = [json.loads(line) for line in
                     (self.output / "annotations" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(notes), count)
            for note in notes:
                image = self.output / note["image"]
                self.assertTrue(image.is_file())
                lines = (self.output / "labels" / split / f"{image.stem}.txt").read_text().splitlines()
                self.assertEqual(len(lines), len(note["items"]))
                for line, item in zip(lines, note["items"]):
                    cls, *values = line.split()
                    self.assertEqual(CLASS_NAMES[int(cls)], item["class"])
                    self.assertEqual(CHAR[item["class"]], item["char"])
                    self.assertTrue(all(0 <= float(v) <= 1 for v in values))
                    self.assertLessEqual(abs(item["angle"]), 45.0)
                    self.assertTrue(0 < item["visible"] <= 1.0)

    def test_images_include_truncated_and_occluded_jamo(self):
        import random

        rng = random.Random(5)
        backgrounds = hangul_dataset.Backgrounds(rng, video="없는 파일")
        partial = clipped = total = 0
        for _ in range(40):
            image, items = hangul_dataset.make_image(rng, backgrounds, FONTS[:3], 45.0)
            height, width = image.shape[:2]
            for item in items:
                x1, y1, x2, y2 = item["box"]
                self.assertTrue(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)
                self.assertGreater(item["visible"], 0.1)   # 거의 안 보이는 자모는 라벨에 없어야 합니다.
                total += 1
                partial += item["visible"] < 0.95
                clipped += x1 == 0 or y1 == 0 or x2 == width or y2 == height
        self.assertGreater(partial, 0.05 * total)
        self.assertGreater(clipped, 0)

    def test_tracking_sequence(self):
        folder = self.output / "tracking" / "test" / "jamo-01"
        sequence = Sequence(folder)
        self.assertEqual(sequence.length, 40)
        self.assertTrue(sequence.frame_path(40).is_file())
        texts = json.loads((folder / "gt" / "objects.json").read_text(encoding="utf-8"))
        self.assertTrue(all(CHAR[value["class"]] == value["char"] for value in texts.values()))
        chars = [value["char"] for value in texts.values()]
        self.assertLess(len(set(chars)), len(chars))  # 같은 자모가 여러 개 나와야 합니다.
        angles = (folder / "gt" / "angles.csv").read_text().splitlines()[1:]
        self.assertTrue(all(abs(float(line.split(",")[2])) <= 45.0 for line in angles))
        rows = [line.split(",") for line in (folder / "gt" / "gt.txt").read_text().splitlines()]
        self.assertTrue(rows)
        for frame, track_id, x, y, w, h, flag, cls, visibility in rows:
            self.assertIn(track_id, texts)
            self.assertTrue(1 <= int(frame) <= 40)
            self.assertTrue(int(x) >= 1 and int(y) >= 1 and int(w) > 0 and int(h) > 0)  # 1부터 시작하는 좌표
            self.assertTrue(int(x) - 1 + int(w) <= sequence.width and int(y) - 1 + int(h) <= sequence.height)
            self.assertTrue(0 <= float(visibility) <= 1)

    def test_dataset_yaml_and_splits(self):
        yaml = (self.output / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("0: giyeok", yaml)
        self.assertIn("39: ui", yaml)
        splits = json.loads((self.output / "splits.json").read_text(encoding="utf-8"))
        self.assertEqual(splits["counts"]["detection"]["train"]["images"], 6)
        self.assertEqual(len(splits["classes"]), 40)
        self.assertEqual(splits["rotation"], 45.0)

    def test_existing_output_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            hangul_dataset.main(Namespace(output=str(self.output), train=1, val=1, test=1, sequences=1, length=5,
                                          backgrounds=None, fonts=None, seed=1, rotation=45.0))


if __name__ == "__main__":
    unittest.main()
