"""데이터 검사기의 오류 검출·시각 자료·읽기 전용 동작을 실제 파일로 확인합니다."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from inspect_camouflage_data import inspect, main, read_yolo
from tests.test_camouflage import prepare_moca


class DataInspectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.dataset = prepare_moca(self.root)
        self.output = self.root / "report"

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self):
        return json.loads((self.dataset / "splits.json").read_text())

    def write_manifest(self, data):
        (self.dataset / "splits.json").write_text(json.dumps(data), encoding="utf-8")

    def run_inspection(self, **options):
        return inspect(self.dataset, self.output, **options)

    def codes(self, report):
        return {issue["code"] for issue in report["issues"]}

    def label_path(self, row):
        return self.dataset / "labels" / row["split"] / (Path(row["image"]).stem + ".txt")

    def test_valid_dataset_generates_review_material_without_changing_input(self):
        before = {str(p.relative_to(self.dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.dataset.rglob("*") if p.is_file()}
        report = self.run_inspection(samples=3, check_hashes=True)
        self.assertEqual(report["errors"], 0)
        self.assertEqual(report["warnings"], 0)
        self.assertEqual(len(report["sequences"]), 3)
        self.assertTrue(report["hashes_recomputed"])
        self.assertEqual({p["split"] for p in report["previews"]}, {"train", "val", "test"})
        self.assertTrue(all(p["declared_ids"] == [7] for p in report["previews"]))
        self.assertIn("사람", report["status"])
        for preview in report["previews"]:
            with Image.open(self.output / preview["preview"]) as image:
                image.load()
                self.assertEqual(image.size, (80, 60))
                self.assertEqual(image.getpixel((10, 39)), (0, 255, 112))
        page = (self.output / "index.html").read_text()
        self.assertIn("모델 예측 결과가 아닙니다", page)
        self.assertNotIn("<script", page)
        self.assertEqual(json.loads((self.output / "report.json").read_text())["errors"], 0)
        after = {str(p.relative_to(self.dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.dataset.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_invalid_box_and_missing_label_are_errors(self):
        rows = self.manifest()["records"]
        self.label_path(rows[0]).write_text("0 0.98 0.5 0.2 0.5\n")
        self.label_path(rows[1]).unlink()
        report = self.run_inspection()
        self.assertGreaterEqual(report["errors"], 2)
        self.assertIn("record", self.codes(report))
        self.assertIn("구조 오류", report["status"])
        self.assertTrue((self.output / "index.html").exists())

    def test_nonfinite_wrong_class_and_zero_area_are_rejected(self):
        path = self.root / "label.txt"
        for line in ("0 nan 0.5 0.2 0.2", "1 0.5 0.5 0.2 0.2", "0 0.5 0.5 0 0.2"):
            with self.subTest(line=line):
                path.write_text(line)
                with self.assertRaises(ValueError):
                    read_yolo(path)

    def test_hash_changes_and_cross_split_duplicates_are_detected(self):
        manifest = self.manifest()
        first = manifest["records"][0]
        other = next(row for row in manifest["records"] if row["split"] != first["split"])
        # 원본은 하드링크일 수 있으므로 테스트 파일도 unlink 후 대체합니다.
        target = self.dataset / other["image"]
        target.unlink()
        target.write_bytes((self.dataset / first["image"]).read_bytes())
        report = self.run_inspection(check_hashes=True)
        self.assertIn("hash_changed", self.codes(report))
        self.assertIn("image_leak", self.codes(report))

    def test_tracking_ids_boxes_and_unannotated_frames_are_checked(self):
        sequence = next((self.dataset / "tracking").glob("*/*"))
        gt = sequence / "gt" / "gt.txt"
        line = gt.read_text().splitlines()[0]
        gt.write_text(line + "\n" + line + "\n")
        report = self.run_inspection()
        self.assertIn("sequence", self.codes(report))
        self.assertTrue(any("중복" in i["message"] for i in report["issues"]))

    def test_gt_yolo_disagreement_is_reported(self):
        sequence = next((self.dataset / "tracking").glob("*/*"))
        gt = sequence / "gt" / "gt.txt"
        lines = gt.read_text().splitlines()
        fields = lines[0].split(",")
        fields[2] = "12"
        gt.write_text(",".join(fields) + "\n" + "\n".join(lines[1:]) + "\n")
        report = self.run_inspection()
        self.assertIn("gt_label_mismatch", self.codes(report))

    def test_missing_unannotated_frame_is_not_ignored(self):
        sequence = next((self.dataset / "tracking").glob("*/*"))
        (sequence / "img1" / "000002.png").unlink()
        report = self.run_inspection()
        self.assertIn("sequence", self.codes(report))
        self.assertGreater(report["errors"], 0)

    def test_unknown_files_are_reported_without_deletion(self):
        extra = self.dataset / "labels" / "train" / "unused<script>.txt"
        extra.write_text("unused")
        report = self.run_inspection()
        self.assertIn("unlisted_file", self.codes(report))
        self.assertTrue(extra.exists())
        self.assertEqual(report["errors"], 0)
        page = (self.output / "index.html").read_text()
        self.assertIn("unused&lt;script&gt;.txt", page)
        self.assertNotIn("unused<script>", page)

    def test_tracking_only_frame_leak_between_splits_is_detected(self):
        sequences = list((self.dataset / "tracking").glob("*/*"))
        first = sequences[0] / "img1" / "000002.png"
        other = sequences[1] / "img1" / "000002.png"
        other.unlink()
        other.write_bytes(first.read_bytes())
        report = self.run_inspection(check_hashes=True)
        self.assertIn("tracking_image_leak", self.codes(report))

    def test_corrupted_image_and_incorrect_manifest_counts_are_errors(self):
        data = self.manifest()
        path = self.dataset / data["records"][0]["image"]
        path.unlink()
        path.write_bytes(b"not an image")
        data["counts"]["train"]["images"] += 5
        self.write_manifest(data)
        report = self.run_inspection()
        self.assertIn("record", self.codes(report))
        self.assertIn("counts", self.codes(report))
        self.assertGreater(report["errors"], 0)

    def test_ids_not_provided_are_a_warning_not_a_claim_of_verified_identity(self):
        for metadata in (self.dataset / "tracking").rglob("annotations.json"):
            data = json.loads(metadata.read_text())
            data["ids_verified"] = False
            metadata.write_text(json.dumps(data))
            (metadata.parent / "gt" / "gt.txt").unlink()
        manifest = self.manifest()
        for seq in manifest["sequences"]:
            seq["ids_verified"] = False
        self.write_manifest(manifest)
        report = self.run_inspection()
        self.assertEqual(report["errors"], 0)
        self.assertIn("ids_unverified", self.codes(report))
        self.assertTrue(all(not s["ids_declared"] for s in report["sequences"]))
        self.assertIn("개체 ID의 의미", report["limits"][0])

    def test_bad_manifest_and_yaml_still_produce_a_report(self):
        (self.dataset / "dataset.yaml").write_text("names: [\n")
        report = self.run_inspection()
        self.assertIn("yaml", self.codes(report))
        second = self.root / "second"
        (self.dataset / "splits.json").write_text("{")
        report = inspect(self.dataset, second)
        self.assertIn("manifest", self.codes(report))
        self.assertTrue((second / "report.json").exists())

    def test_manifest_cannot_read_files_outside_dataset(self):
        manifest = self.manifest()
        manifest["records"][0]["image"] = "../../outside.png"
        self.write_manifest(manifest)
        report = self.run_inspection()
        self.assertIn("record", self.codes(report))
        self.assertGreater(report["errors"], 0)

    def test_output_does_not_overwrite_or_modify_dataset(self):
        for output in (self.dataset, self.dataset / "inspection", self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                inspect(self.dataset, output)
        self.assertFalse((self.dataset / "inspection").exists())

    def test_cli_exit_code_distinguishes_structural_error_from_success(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = main([str(self.dataset), "--output", str(self.output), "--samples", "0"])
        self.assertEqual(code, 0)
        self.assertEqual(list(self.output.glob("preview_*.png")), [])
        self.label_path(self.manifest()["records"][0]).unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            code = main([str(self.dataset), "--output", str(self.root / "bad")])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
