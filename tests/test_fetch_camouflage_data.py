"""다운로드 도구의 표본·좌표·무손실 마스크·부분 읽기 검사. 네트워크 없이 실행합니다."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from PIL import Image

import fetch_camouflage_data as fetch


class Response(io.BytesIO):
    def __init__(self, data, status, headers):
        super().__init__(data)
        self.status, self.headers = status, headers


class AcquisitionTest(unittest.TestCase):
    def test_http_ranges_seek_and_wrong_response_are_checked(self):
        payload = b"abcdef"

        def server(request, **kwargs):
            if request.get_method() == "HEAD":
                return Response(b"", 200, {"Content-Length": "6"})
            start, end = map(int, request.get_header("Range").split("=")[1].split("-"))
            return Response(payload[start:end+1], 206, {"Content-Range": f"bytes {start}-{end}/6"})

        with mock.patch("urllib.request.urlopen", side_effect=server):
            reader = fetch.HTTPRangeReader("https://example.test/data.zip")
            self.assertEqual(reader.read(2), b"ab")
            reader.seek(-2, 2)
            self.assertEqual(reader.read(), b"ef")
            self.assertEqual(reader.read(1), b"")
            reader.seek(0)
            with mock.patch("urllib.request.urlopen", return_value=Response(payload, 200, {})):
                with self.assertRaises(ValueError):
                    reader.read(2)
            with self.assertRaises(ValueError):
                reader.seek(-1)

    def test_original_via_only_existing_frames_no_invented_ids(self):
        text = '# notes\n# CSV_HEADER = metadata_id,file_list,flags,temporal_coordinates,spatial_coordinates,metadata\n'
        text += 'a,/crab/00000.jpg,0,[],"[2,1,2,3,4]",{}\n'
        text += 'b,/crab/00005.jpg,0,[],"[2,2,3,4,5]",{}\n'
        rows = fetch.moca_rows(text, ["crab"], {"crab": ["00000.jpg"]})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["track_id"], "")
        self.assertEqual((rows[0]["x"], rows[0]["height"]), (1, 4))
        with self.assertRaises(ValueError):
            fetch.moca_rows("no header", ["crab"], {"crab": []})

    def test_short_video_uses_available_frames_instead_of_guessing_length(self):
        names = ["crab", "fox", "moth"]
        image = io.BytesIO()
        Image.new("RGB", (8, 6), "green").save(image, format="JPEG")
        text = '# CSV_HEADER = metadata_id,file_list,flags,temporal_coordinates,spatial_coordinates,metadata\n'
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in names:
                text += f'a,/{name}/00000.jpg,0,[],"[2,0,0,8,6]",{{}}\n'
                for number in range(2):
                    archive.writestr(f"MoCA/JPEGImages/{name}/{number:05d}.jpg", image.getvalue())
            archive.writestr("MoCA/Annotations/annotations.csv", text)
        reader = io.BytesIO(buffer.getvalue())
        reader.size, reader.transferred = len(buffer.getvalue()), 0
        with tempfile.TemporaryDirectory() as folder, mock.patch("fetch_camouflage_data.HTTPRangeReader", return_value=reader):
            result = fetch.fetch_moca(Path(folder)/"MoCA", names, 60)
            self.assertEqual(len(result["records"]), 6)
            self.assertEqual(result["annotated_boxes"], 3)

    def test_cod_original_bytes_and_png_mask_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            rgb, mask = io.BytesIO(), io.BytesIO()
            Image.new("RGB", (8, 6), "green").save(rgb, format="JPEG")
            image = Image.new("L", (8, 6))
            image.putpixel((2, 2), 255)
            image.save(mask, format="PNG")
            row = dict(image=dict(path="COD10K-CAM-test.jpg", bytes=rgb.getvalue()),
                       mask=dict(path="COD10K-CAM-test.png", bytes=mask.getvalue()))
            note = fetch.save_cod_pair(row, Path(folder), "Train")
            self.assertEqual((Path(folder)/"Train/GT_Object/COD10K-CAM-test.png").read_bytes(), mask.getvalue())
            self.assertEqual(len(note["image_sha256"]), 64)
            with self.assertRaises(ValueError):
                fetch.save_cod_pair(row, Path(folder), "Train")

    def test_full_mode_checks_disk_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            args = fetch.parser().parse_args(["--output", str(Path(folder)/"raw"), "--full"])
            with mock.patch("shutil.disk_usage", return_value=mock.Mock(free=1024)), \
                    mock.patch("fetch_camouflage_data.fetch_cod") as download:
                with self.assertRaisesRegex(ValueError, "40GiB"):
                    fetch.acquire(args)
                download.assert_not_called()

    def test_failure_does_not_publish_partial_data(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"raw"
            args = fetch.parser().parse_args(["--output", str(target)])
            with mock.patch("fetch_camouflage_data.fetch_cod", return_value={}), \
                    mock.patch("fetch_camouflage_data.fetch_moca", side_effect=RuntimeError("다운로드 실패")):
                with self.assertRaises(RuntimeError):
                    fetch.acquire(args)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
