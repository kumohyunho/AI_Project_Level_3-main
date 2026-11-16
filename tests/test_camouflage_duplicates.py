"""실제 Colab 오류와 같은 Train/Test 동일 바이트 입력의 누수 방지 회귀 검사."""
import contextlib
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import camouflage_dataset as dataset
from inspect_camouflage_data import inspect
from tests.test_camouflage import image


class DuplicatePolicyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for split, count, offset in (("Train", 8, 40), ("Test", 2, 100)):
            for index in range(count):
                name = f"COD10K-CAM-{split}-{index}"
                image(self.root/split/"Image"/f"{name}.png", offset+index)
                path = self.root/split/"GT_Object"/f"{name}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                mask = np.zeros((60, 80), np.uint8)
                mask[10:40, 20:60] = 255
                Image.fromarray(mask).save(path)
        self.original = self.root/"Train/Image/COD10K-CAM-Train-0.png"
        self.duplicate = self.root/"Test/Image/COD10K-CAM-Test-0.png"
        shutil.copyfile(self.original, self.duplicate)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, output, policy="error"):
        args = dataset.parser().parse_args(["--cod10k", str(self.root), "--output", str(output),
                                            "--cod10k-duplicates", policy])
        with contextlib.redirect_stdout(io.StringIO()):
            dataset.prepare(args)
        return json.loads((output/"splits.json").read_text())

    def test_default_still_rejects_leak_without_partial_output(self):
        output = self.root/"strict"
        with self.assertRaisesRegex(ValueError, "중복"):
            self.prepare(output)
        self.assertFalse(output.exists())

    def test_prefer_test_retains_test_excludes_training_and_passes_inspector(self):
        before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for split in ("Train", "Test")
                  for p in (self.root/split).rglob("*") if p.is_file()}
        output = self.root/"prepared"
        data = self.prepare(output, "prefer-test")
        details = data["cod10k_duplicates"]
        self.assertEqual(details["excluded_train_count"], 1)
        self.assertEqual(details["retained_test_count"], 2)
        self.assertEqual(details["excluded_train"][0]["original"], str(self.original))
        records = data["records"]
        self.assertFalse(any(r["original"] == str(self.original) for r in records))
        self.assertTrue(any(r["original"] == str(self.duplicate) and r["split"] == "test" for r in records))
        report = inspect(output, self.root/"review", check_hashes=True)
        self.assertEqual(report["errors"], 0)
        after = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for split in ("Train", "Test")
                 for p in (self.root/split).rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_train_duplicates_stay_in_same_split_and_selection_is_repeatable(self):
        original = self.root/"Train/Image/COD10K-CAM-Train-1.png"
        duplicate = self.root/"Train/Image/COD10K-CAM-Train-2.png"
        shutil.copyfile(original, duplicate)
        first = self.prepare(self.root/"first", "prefer-test")
        second = self.prepare(self.root/"second", "prefer-test")
        self.assertEqual(first["records"], second["records"])
        rows = [r for r in first["records"] if r["original"] in (str(original), str(duplicate))]
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({r["split"] for r in rows}), 1)
        self.assertEqual(len(first["cod10k_duplicates"]["train_repeated_hash_groups"]), 1)

    def test_too_few_remaining_groups_fails_instead_of_leaking_into_val(self):
        for p in (self.root/"Train/Image").glob("*.png"):
            if p != self.original:
                shutil.copyfile(self.duplicate, p)
        output = self.root/"empty"
        with self.assertRaises(ValueError):
            self.prepare(output, "prefer-test")
        self.assertFalse(output.exists())

    def test_all_official_test_records_preserved_when_test_itself_has_duplicates(self):
        other = self.root/"Test/Image/COD10K-CAM-Test-1.png"
        shutil.copyfile(self.duplicate, other)
        data = self.prepare(self.root/"prepared", "prefer-test")
        self.assertEqual(data["counts"]["test"]["images"], 2)
        detail = data["cod10k_duplicates"]
        self.assertEqual(len(detail["test_repeated_hash_groups"]), 1)
        self.assertEqual(len(detail["excluded_train"][0]["kept_test_files"]), 2)


if __name__ == "__main__":
    unittest.main()
