"""한글 합성 데이터 생성기 테스트 (한글 글꼴이 없으면 건너뜁니다).  실행: ``python -m unittest discover -s tests -t .``"""
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hangul_dataset  # noqa: E402
from mot_data import Sequence  # noqa: E402

FONTS = hangul_dataset.find_fonts()


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
                                      backgrounds=None, fonts=None, seed=1))

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
                for line in lines:
                    cls, *values = line.split()
                    self.assertEqual(cls, "0")
                    self.assertTrue(all(0 <= float(v) <= 1 for v in values))
                self.assertTrue(all(item["text"].strip() for item in note["items"]))

    def test_tracking_sequence(self):
        folder = self.output / "tracking" / "test" / "hangul-01"
        sequence = Sequence(folder)
        self.assertEqual(sequence.length, 40)
        self.assertTrue(sequence.frame_path(40).is_file())
        texts = json.loads((folder / "gt" / "texts.json").read_text(encoding="utf-8"))
        rows = [line.split(",") for line in (folder / "gt" / "gt.txt").read_text().splitlines()]
        self.assertTrue(rows)
        for frame, track_id, x, y, w, h, flag, cls, visibility in rows:
            self.assertIn(track_id, texts)
            self.assertTrue(1 <= int(frame) <= 40)
            self.assertTrue(int(x) >= 0 and int(y) >= 0 and int(w) > 0 and int(h) > 0)
            self.assertTrue(int(x) + int(w) <= sequence.width and int(y) + int(h) <= sequence.height)
            self.assertTrue(0 <= float(visibility) <= 1)

    def test_dataset_yaml_and_splits(self):
        yaml = (self.output / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("names: {0: text}", yaml)
        splits = json.loads((self.output / "splits.json").read_text(encoding="utf-8"))
        self.assertEqual(splits["counts"]["detection"]["train"]["images"], 6)

    def test_existing_output_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            hangul_dataset.main(Namespace(output=str(self.output), train=1, val=1, test=1, sequences=1, length=5,
                                          backgrounds=None, fonts=None, seed=1))


if __name__ == "__main__":
    unittest.main()
