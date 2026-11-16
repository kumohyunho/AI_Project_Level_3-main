"""실제 COD10K·MoCA 확보. 기본은 표본, --full은 Colab 전체 데이터 준비용입니다.

COD10K는 nobg 공개 미러의 원본 이미지/PNG 마스크를 사용하고 revision을 고정합니다.
MoCA는 Oxford 공식 ZIP에서 선택한 원본 프레임·수동 박스만 부분 다운로드합니다.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import urllib.request
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

MOCA_URL = "https://thor.robots.ox.ac.uk/datasets/MoCA/MoCA.zip"
COD_REPO = "nobg/COD10K"


class HTTPRangeReader(io.RawIOBase):
    """ZIP/Parquet의 요청 구간만 읽습니다. 서버가 Range를 무시하면 중단합니다."""
    def __init__(self, url):
        super().__init__()
        self.url, self.position, self.transferred = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("잘못된 seek 기준입니다.")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError("음수 파일 위치입니다.")
        self.position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        size = min(self.size-self.position, size if size >= 0 else self.size-self.position)
        if size <= 0:
            return b""
        if size > 512*1024*1024:
            raise ValueError("단일 요청은 512MiB 이하로 제한합니다.")
        start, end = self.position, self.position+size-1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise ValueError("서버가 요청한 바이트 구간을 정확하게 반환하지 않았습니다.")
            data = response.read(size+1)
        if len(data) != size:
            raise ValueError("부분 다운로드 길이가 일치하지 않습니다.")
        self.position += size
        self.transferred += size
        return data


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_cod_pair(row, root, official_split):
    """미러 원본 bytes를 저장합니다. JPEG 미리보기 마스크를 정답으로 사용하지 않습니다."""
    name = Path(row["image"]["path"]).name
    mask_name = Path(row["mask"]["path"]).name
    if not name.startswith("COD10K-CAM-") or Path(name).stem != Path(mask_name).stem:
        raise ValueError("COD10K 원본 이름/이미지·마스크 쌍이 맞지 않습니다.")
    image_bytes, mask_bytes = row["image"]["bytes"], row["mask"]["bytes"]
    with Image.open(io.BytesIO(image_bytes)) as image, Image.open(io.BytesIO(mask_bytes)) as mask:
        if image.size != mask.size or mask.format != "PNG":
            raise ValueError("동일 크기 이미지와 무손실 PNG 마스크가 필요합니다.")
        values = set(np.unique(np.array(mask)).tolist())
        if not values <= {0, 1, 255} or not any(v != 0 for v in values):
            raise ValueError("COD10K 이진 전경 마스크가 아닙니다.")
    for folder, filename, data in (("Image", name, image_bytes), ("GT_Object", Path(name).stem+".png", mask_bytes)):
        target = root/official_split/folder/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f"중복 원본 파일: {name}")
        target.write_bytes(data)
    return dict(source="cod10k", official_split=official_split, file=name,
                image_sha256=digest(image_bytes), mask_sha256=digest(mask_bytes))


def fetch_cod(root, train_count, test_count, seed, full=False):
    import pyarrow.parquet as parquet

    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{COD_REPO}", timeout=30) as response:
        revision = json.load(response)["sha"]
    records, downloads = [], []
    for split, count, official in (("train", train_count, "Train"), ("test", test_count, "Test")):
        for shard in range(2):
            url = f"https://huggingface.co/datasets/{COD_REPO}/resolve/{revision}/data/{split}-{shard:05d}-of-00002.parquet"
            with HTTPRangeReader(url) as remote:
                source = parquet.ParquetFile(remote)
                take = 0
                for group in range(source.metadata.num_row_groups if full else 1):
                    rows = source.read_row_group(group).to_pylist()
                    rows.sort(key=lambda row: digest(f"{seed}:{row['image']['path']}".encode()))
                    requested = len(rows) if full else count//2 + (1 if shard < count % 2 else 0)
                    if requested > len(rows):
                        raise ValueError("요청 표본 수가 첫 row group 크기보다 큽니다.")
                    for row in rows[:requested]:
                        records.append(save_cod_pair(row, root, official))
                    take += requested
                downloads.append(dict(url=url, file_bytes=remote.size, transferred_bytes=remote.transferred,
                                      split=split, shard=shard, row_groups="all" if full else [0], sampled=take))
                print(f"COD10K {split} shard {shard}: {take}쌍 확보", flush=True)
    return dict(repository=COD_REPO, revision=revision, downloads=downloads,
                sampling="미러의 train/test 전체 보호색 이미지 (Background·NonCAM 미포함)" if full else
                         "각 shard 첫 row group에서 원본 파일명+seed SHA256 순서로 선택. 전체 대표성 보장 안 함.",
                mirror_card=f"https://huggingface.co/datasets/{COD_REPO}/blob/{revision}/README.md", records=records)


def moca_rows(text, names, available):
    """공식 VIA CSV의 주석 헤더를 읽고 원본의 수동 사각형 행만 정규화합니다."""
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith("# CSV_HEADER = ")), None)
    if header is None:
        raise ValueError("공식 MoCA VIA CSV 헤더를 찾을 수 없습니다.")
    result = []
    for row in csv.reader(lines[header+1:]):
        if not row or not row[0].strip():
            continue
        parts = row[1].strip("/").split("/")
        if len(parts) != 2 or parts[0] not in names:
            continue
        name, filename = parts
        if filename not in available[name]:
            continue
        shape = json.loads(row[4])
        if len(shape) != 5 or shape[0] != 2 or not all(math.isfinite(v) for v in shape[1:]) or min(shape[3:]) <= 0:
            raise ValueError("지원하지 않거나 잘못된 MoCA 사각형 주석입니다.")
        result.append(dict(sequence=name, filename=filename, x=shape[1], y=shape[2], width=shape[3], height=shape[4], track_id=""))
    if set(names) != {r["sequence"] for r in result}:
        raise ValueError("선택한 영상 중 주석이 없는 영상이 있습니다.")
    return result


def download_archive(path):
    checksum = hashlib.sha256()
    transferred = 0
    with urllib.request.urlopen(MOCA_URL, timeout=120) as response, path.open("wb") as target:
        expected = int(response.headers["Content-Length"])
        while chunk := response.read(4*1024*1024):
            target.write(chunk)
            checksum.update(chunk)
            transferred += len(chunk)
            if transferred % (256*1024*1024) < len(chunk):
                print(f"MoCA ZIP: {transferred/1024**3:.1f}/{expected/1024**3:.1f} GiB", flush=True)
    if transferred != expected:
        raise ValueError("MoCA ZIP 다운로드 길이가 맞지 않습니다.")
    return checksum.hexdigest()


def fetch_moca(root, names, frames, full=False):
    if any(Path(name).name != name or name in ("", ".", "..") for name in names):
        raise ValueError("MoCA 영상 이름이 올바르지 않습니다.")
    records, archive_hash = [], None
    root.mkdir(parents=True)
    downloaded = root.parent/"MoCA-download.zip"
    if full:
        archive_hash = download_archive(downloaded)
    with (downloaded.open("rb") if full else HTTPRangeReader(MOCA_URL)) as remote, zipfile.ZipFile(remote) as archive:
        members = {}
        for member in archive.namelist():
            parts = member.split("/")
            if len(parts) == 4 and parts[:2] == ["MoCA", "JPEGImages"] and parts[3].endswith(".jpg"):
                members.setdefault(parts[2], []).append(parts[3])
        if full:
            names = sorted(members)
        if any(Path(name).name != name or name in ("", ".", "..") for name in names):
            raise ValueError("압축 파일의 영상 이름이 올바르지 않습니다.")
        available = {}
        for name in names:
            files = sorted(members[name], key=lambda s: int(Path(s).stem))
            available[name] = files if full else files[:frames]
        annotations = archive.read("MoCA/Annotations/annotations.csv")
        (root/"annotations-original.csv").write_bytes(annotations)
        rows = moca_rows(annotations.decode("utf-8-sig"), names, available)
        for name in names:
            for filename in available[name]:
                member = f"MoCA/JPEGImages/{name}/{filename}"
                data = archive.read(member)  # ZipFile가 CRC도 확인합니다. 전체 ZIP은 저장하지 않습니다.
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                target = root/"frames"/name/filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                records.append(dict(source="moca", file=f"{name}/{filename}", sha256=digest(data)))
            print(f"MoCA {name}: {len(available[name])}프레임 확보", flush=True)
        with (root/"boxes.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        info = dict(url=MOCA_URL, archive_bytes=downloaded.stat().st_size if full else remote.size,
                    transferred_bytes=downloaded.stat().st_size if full else remote.transferred, archive_sha256=archive_hash,
                    annotation_sha256=digest(annotations), annotated_boxes=len(rows), records=records,
                    ids="원본에 추적 ID 없음. 임의 ID를 만들지 않음.",
                    fps="프레임 이미지에서 FPS 확인 불가. 전처리의 24fps는 검출 학습용 임시값이며 시간 성능 분석에 사용 안 함.")
    if full:
        downloaded.unlink()
    return info


def acquire(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"출력 폴더가 이미 있습니다: {output}")
    if (min(args.train, args.test, args.frames) < 1 or len(set(args.sequences)) < 3
            or len(set(args.sequences)) != len(args.sequences)):
        raise ValueError("양의 표본 수와 서로 다른 MoCA 영상 최소 3개가 필요합니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.full and shutil.disk_usage(output.parent).free < 40*1024**3:
        raise ValueError("전체 데이터 준비에는 먼저 여유 공간 40GiB 이상을 확보하세요.")
    with TemporaryDirectory(dir=output.parent) as folder:
        staging = Path(folder)/"raw"
        staging.mkdir()
        cod = fetch_cod(staging/"COD10K-v3", args.train, args.test, args.seed, args.full)
        moca = fetch_moca(staging/"MoCA", args.sequences, args.frames, args.full)
        manifest = dict(seed=args.seed, full=args.full, cod10k=cod, moca=moca,
                        scope="COD10K 미러의 보호색 이미지 전체 + MoCA 전체" if args.full else
                              "실제 공개 데이터의 첫 학습용 소규모 표본. 전체 데이터·성능 최적화 아님.")
        (staging/"acquisition.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.rename(output)
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("datasets/raw/camouflage-first"))
    p.add_argument("--train", type=int, default=256)
    p.add_argument("--test", type=int, default=64)
    p.add_argument("--sequences", nargs="+", default=["chameleon", "crab", "arctic_fox"])
    p.add_argument("--frames", type=int, default=30, help="영상당 최대 표본 프레임 수, 짧은 영상은 있는 프레임만 사용")
    p.add_argument("--full", action="store_true", help="Colab 등 40GiB 이상 여유 공간에서 전체 데이터 준비")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    acquire(parser().parse_args())
