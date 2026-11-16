"""COD10K 마스크·MoCA 박스 CSV → YOLO 학습 데이터와 추적 평가 시퀀스.

실행·원본 폴더·CSV 형식: docs/CAMOUFLAGE.md.
MoCA는 영상 단위로 분리하며, ID가 없으면 추적 정답을 만들지 않습니다.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from extensions.camouflage import CLASS_NAME

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def image_files(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"이미지가 없습니다: {folder}")
    if len({p.stem for p in files}) != len(files):
        raise ValueError(f"확장자만 다른 중복 이미지 이름이 있습니다: {folder}")
    return files


def mask_boxes(path, instance=False):
    """이진 마스크는 전경 전체 박스, 인스턴스 마스크는 0 이외 값/색별 박스.

    연결 성분을 개체로 가정하지 않습니다. 분리된 신체도 같은 박스에 포함합니다.
    RGB 인스턴스 마스크는 검정을 배경으로 봅니다.
    """
    with Image.open(path) as image:
        mask = np.array(image)
    if mask.ndim == 3:
        mask = mask[:, :, :3].astype(np.int64)
        mask = mask[:, :, 0] * 65536 + mask[:, :, 1] * 256 + mask[:, :, 2]
    boxes = []
    for value in np.unique(mask) if instance else [None]:
        if value == 0:
            continue
        ys, xs = np.nonzero(mask == value if instance else mask != 0)
        if len(xs):
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes, (mask.shape[1], mask.shape[0])


def clipped_box(values, width, height):
    x, y, w, h = map(float, values)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("박스는 유한한 좌표와 양의 너비·높이를 가져야 합니다.")
    box = (max(0.0, x), max(0.0, y), min(float(width), x + w), min(float(height), y + h))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("영상 밖에 있는 박스입니다.")
    return box


def partition(names, val_ratio, test_ratio, seed):
    """재현 가능한 분할. MoCA에는 프레임이 아닌 영상 이름을 넣습니다."""
    names = sorted(names, key=lambda n: hashlib.sha256(f"{seed}:{n}".encode()).hexdigest())
    required = 1 + int(val_ratio > 0) + int(test_ratio > 0)
    if len(names) < required:
        raise ValueError(f"분할하려면 최소 {required}개 항목(영상)이 필요합니다.")
    nval = max(1, int(len(names) * val_ratio)) if val_ratio else 0
    ntest = max(1, int(len(names) * test_ratio)) if test_ratio else 0
    return {name: ("val" if i < nval else "test" if i < nval + ntest else "train")
            for i, name in enumerate(names)}


def read_moca_csv(path, single_object=False):
    """정규화 CSV 또는 MoCA 연구 코드의 file_list/spatial_coordinates CSV.

    정규화: sequence,filename,x,y,width,height,track_id. 좌표는 0 기반 xywh.
    VIA 파생: file_list=/영상/00000.jpg, spatial_coordinates=[2,x,y,w,h].
    ID 없음은 None. --single-object는 사용자가 개체 한 마리임을 확인했다는 뜻입니다.
    좌표 네 칸이 모두 비면 명시적인 빈 정답 프레임입니다.
    """
    sequences = defaultdict(lambda: defaultdict(list))
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        via = {"file_list", "spatial_coordinates"} <= set(reader.fieldnames or [])
        required = {"sequence", "filename", "x", "y", "width", "height"}
        if not via and not required <= set(reader.fieldnames or []):
            raise ValueError("지원하는 MoCA CSV 열이 없습니다. docs/CAMOUFLAGE.md를 확인하세요.")
        for row in reader:
            if via:
                relative = Path(row["file_list"].replace("\\", "/").lstrip("/"))
                if len(relative.parts) != 2:
                    raise ValueError("file_list는 /영상이름/프레임파일 형식이어야 합니다.")
                sequence, filename = relative.parts
                shape = json.loads(row["spatial_coordinates"])
                if len(shape) != 5 or shape[0] != 2:
                    raise ValueError("MoCA 사각형 [2,x,y,width,height]만 지원합니다.")
                values = shape[1:]
            else:
                sequence, filename = row["sequence"], row["filename"]
                values = [row[k] for k in ("x", "y", "width", "height")]
            if any(Path(v).name != v or v in ("", ".", "..") or "\\" in v
                   for v in (sequence, filename)):
                raise ValueError("영상 이름과 프레임 파일은 하위 경로를 포함하면 안 됩니다.")
            items = sequences[sequence][filename]
            if all(v == "" for v in values):
                if items:
                    raise ValueError("빈 정답과 객체 정답이 같은 프레임에 있습니다.")
                items.append(None)
                continue
            if None in items:
                raise ValueError("빈 정답과 객체 정답이 같은 프레임에 있습니다.")
            raw_id = (row.get("track_id") or "").strip()
            track_id = int(raw_id) if raw_id else 1 if single_object else None
            if track_id is not None and track_id < 1:
                raise ValueError("track_id는 1 이상의 정수여야 합니다.")
            if single_object and (items or (track_id is not None and track_id != 1)):
                raise ValueError("단일 개체 확인과 맞지 않는 정답입니다.")
            if track_id is not None and any(item[1] == track_id for item in items):
                raise ValueError("같은 프레임에 중복 track_id가 있습니다.")
            items.append((list(map(float, values)), track_id))
    if not sequences:
        raise ValueError("MoCA 정답이 비어 있습니다.")
    return sequences


def linked_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


class Exporter:
    def __init__(self, staging):
        self.root = staging
        self.manifest = []
        self.hash_splits = {}

    def detection(self, image, boxes, split, name, source, sequence=None):
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        previous = self.hash_splits.setdefault(digest, split)
        if previous != split:
            raise ValueError(f"동일 이미지가 {previous}/{split}에 중복됩니다: {image}")
        with Image.open(image) as im:
            width, height = im.size
        target = self.root / "images" / split / f"{name}{image.suffix.lower()}"
        if target.exists():
            raise ValueError(f"출력 이미지 이름이 중복됩니다: {target.name}")
        linked_copy(image, target)
        label = self.root / "labels" / split / f"{name}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} "
                 f"{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}\n" for x1, y1, x2, y2 in boxes]
        label.write_text("".join(lines), encoding="utf-8")
        self.manifest.append(dict(image=str(target.relative_to(self.root)), original=str(image.resolve()),
                                  split=split, source=source, sequence=sequence, boxes=len(boxes), sha256=digest))


def export_cod10k(exporter, root, mask_folder, instance, seed, val_ratio, duplicate_policy="error"):
    if duplicate_policy not in ("error", "prefer-test"):
        raise ValueError("COD10K 중복 정책을 확인하세요.")
    inputs = {}
    for official in ("Train", "Test"):
        inputs[official] = [p for p in image_files(root / official / "Image")
                            if p.stem.startswith(("COD10K-CAM-", "COD10K-Background-"))]
        if not inputs[official]:
            raise ValueError(f"COD10K CAM/Background 파일이 없습니다: {official}")
    excluded, groups, assignments = [], {}, {}
    test_hashes = defaultdict(list)
    if duplicate_policy == "prefer-test":
        # 공식 Test를 학습에 넣지 않습니다. 원본은 보존하고 생성 데이터에서만 Train 사본을 제외합니다.
        for image in inputs["Test"]:
            test_hashes[hashlib.sha256(image.read_bytes()).hexdigest()].append(image.name)
        for image in inputs["Train"]:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in test_hashes:
                excluded.append(dict(original=str(image.resolve()), sha256=digest,
                                     kept_test_files=test_hashes[digest], reason="공식 Test와 동일 바이트 이미지"))
            else:
                groups.setdefault(digest, []).append(image)
        # Train 내부의 동일 이미지도 같은 그룹으로 배치해 train/val 누수를 막습니다.
        hash_splits = partition(groups, val_ratio, 0, seed)
        assignments = {image.stem:hash_splits[digest] for digest, images in groups.items() for image in images}
        inputs["Train"] = [image for images in groups.values() for image in images]
    for official, split in (("Train", None), ("Test", "test")):
        images = inputs[official]
        # NonCAM은 보호색 양성으로 학습하지 않습니다. CAM·Background만 사용합니다.
        if split is None and duplicate_policy == "error":
            assignments = partition([p.stem for p in images], val_ratio, 0, seed)
        for image in images:
            boxes, size = mask_boxes(root / official / mask_folder / f"{image.stem}.png", instance)
            with Image.open(image) as im:
                if im.size != size:
                    raise ValueError(f"이미지와 마스크 크기가 다릅니다: {image}")
            background = image.stem.startswith("COD10K-Background-")
            if (background and boxes) or (not background and not boxes):
                raise ValueError(f"파일 종류와 전경 마스크가 맞지 않습니다: {image}")
            exporter.detection(image, boxes, split or assignments[image.stem], f"cod_{image.stem}", "cod10k")
    report = dict(policy=duplicate_policy, excluded_train=excluded, excluded_train_count=len(excluded),
                  retained_test_count=len(inputs["Test"]),
                  train_repeated_hash_groups={digest:[p.name for p in images] for digest, images in groups.items() if len(images)>1},
                  test_repeated_hash_groups={digest:names for digest, names in test_hashes.items() if len(names)>1},
                  note="원본 보존. prefer-test는 Test 우선·Train 해시 그룹 분할. 같은 분할 안의 반복은 보존하며 유사 이미지 중복은 별도 확인.")
    print(f"COD10K 중복 정책: {duplicate_policy}, Train/Test 중복으로 제외한 Train 이미지 {len(excluded)}개")
    return report


def export_moca(exporter, frames_root, annotations, single_object, seed, fps):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("MoCA FPS는 양수여야 합니다.")
    sequences = read_moca_csv(annotations, single_object)
    assignments = partition(sequences, 0.15, 0.15, seed)
    summaries = []
    for name, annotated in sorted(sequences.items()):
        split = assignments[name]
        images = image_files(frames_root / name)
        if not all(p.stem.isdigit() for p in images):
            raise ValueError("MoCA 프레임 이름은 숫자여야 합니다.")
        images.sort(key=lambda p: int(p.stem))
        numbers = [int(p.stem) for p in images]
        if any(b != a + 1 for a, b in zip(numbers, numbers[1:])):
            raise ValueError("MoCA에는 연속 프레임이 필요합니다. 정답 없는 프레임도 포함하세요.")
        if len({p.suffix.lower() for p in images}) != 1:
            raise ValueError(f"영상 안의 이미지 확장자가 다릅니다: {name}")
        if set(annotated) - {p.name for p in images}:
            raise FileNotFoundError(f"정답에 해당하는 프레임 파일이 없습니다: {name}")
        verified = all(item is None or item[1] is not None for rows in annotated.values() for item in rows)
        target = exporter.root / "tracking" / split / name
        gt_lines, evaluated, mapping = [], [], {}
        size = None
        for number, image in enumerate(images, 1):
            with Image.open(image) as im:
                if size is not None and im.size != size:
                    raise ValueError(f"영상 안의 프레임 크기가 다릅니다: {name}")
                size = im.size
            linked_copy(image, target / "img1" / f"{number:06d}{image.suffix.lower()}")
            mapping[str(number)] = image.name
            if image.name not in annotated:
                continue  # 미주석 프레임은 음성 학습 데이터도, 평가 정답도 아닙니다.
            evaluated.append(number)
            boxes = []
            for item in annotated[image.name]:
                if item is None:
                    continue
                values, track_id = item
                x1, y1, x2, y2 = clipped_box(values, *size)
                boxes.append((x1, y1, x2, y2))
                if verified:
                    gt_lines.append(f"{number},{track_id},{x1+1:.4f},{y1+1:.4f},"
                                    f"{x2-x1:.4f},{y2-y1:.4f},1,1,1\n")
            exporter.detection(image, boxes, split, f"moca_{name}_{number:06d}", "moca", name)
        width, height = size
        (target / "seqinfo.ini").write_text(
            f"[Sequence]\nname={name}\nimDir=img1\nframeRate={fps}\nseqLength={len(images)}\n"
            f"imWidth={width}\nimHeight={height}\nimExt={images[0].suffix.lower()}\n", encoding="utf-8")
        if verified:
            (target / "gt").mkdir()
            (target / "gt" / "gt.txt").write_text("".join(gt_lines), encoding="utf-8")
        info = dict(sequence=name, split=split, frames=len(images), evaluated_frames=evaluated,
                    ids_verified=verified, single_object_confirmed=single_object, original_files=mapping,
                    note="미주석 프레임은 평가 제외. 입력 주석의 보간 여부는 사용자가 확인.")
        (target / "annotations.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append({k: v for k, v in info.items() if k != "original_files"})
    return summaries


def prepare(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"출력 폴더가 이미 있습니다: {output}")
    if not 0 < args.val_ratio < 1:
        raise ValueError("검증 비율은 0과 1 사이여야 합니다.")
    if not args.cod10k and not args.moca_frames:
        raise ValueError("--cod10k 또는 --moca-frames를 지정하세요.")
    if bool(args.moca_frames) != bool(args.moca_csv):
        raise ValueError("--moca-frames와 --moca-csv를 함께 지정하세요.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as folder:
        staging = Path(folder) / "dataset"
        staging.mkdir()
        exporter = Exporter(staging)
        cod10k_duplicates = None
        if args.cod10k:
            cod10k_duplicates = export_cod10k(exporter, Path(args.cod10k), args.mask_folder, args.instance_masks,
                                            args.seed, args.val_ratio, args.cod10k_duplicates)
        sequences = export_moca(exporter, Path(args.moca_frames), args.moca_csv,
                                args.single_object, args.seed, args.fps) if args.moca_frames else []
        counts = {}
        for split in ("train", "val", "test"):
            rows = [r for r in exporter.manifest if r["split"] == split]
            if not rows:
                raise ValueError(f"{split} 분할에 주석 이미지가 없습니다.")
            counts[split] = dict(images=len(rows), boxes=sum(r["boxes"] for r in rows))
        metadata = dict(classes=[CLASS_NAME], seed=args.seed, counts=counts, sequences=sequences,
                        cod10k_mask_mode="instance" if args.instance_masks else "foreground_union",
                        cod10k_val_ratio=args.val_ratio, moca_split="영상 단위 70/15/15 (최소 1개씩 확보)",
                        cod10k_duplicates=cod10k_duplicates, records=exporter.manifest)
        (staging / "splits.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (staging / "dataset.yaml").write_text(
            f"path: {json.dumps(str(output), ensure_ascii=False)}\ntrain: images/train\nval: images/val\n"
            f"test: images/test\nnames:\n  0: {CLASS_NAME}\n", encoding="utf-8")
        staging.rename(output)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cod10k", type=Path, help="Train/Image·Test/Image가 있는 COD10K 루트")
    p.add_argument("--mask-folder", default="GT_Object")
    p.add_argument("--instance-masks", action="store_true", help="마스크의 0 이외 값/색을 각각 개체로 취급")
    p.add_argument("--cod10k-duplicates", choices=("error", "prefer-test"), default="error",
                   help="기본은 중복 시 중단. prefer-test는 Test 우선 보존·겹치는 Train 제외·Train 해시 그룹 분할")
    p.add_argument("--moca-frames", type=Path, help="영상이름/00000.jpg가 있는 프레임 루트")
    p.add_argument("--moca-csv", type=Path)
    p.add_argument("--single-object", action="store_true", help="모든 입력 영상이 개체 한 마리임을 직접 확인했을 때만 사용")
    p.add_argument("--fps", type=float, default=24.0, help="추출 프레임의 실제 FPS")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1, help="COD10K 공식 Train에서 분리할 검증 비율")
    p.add_argument("--output", type=Path, default=Path("datasets/camouflage"))
    return p


if __name__ == "__main__":
    prepare(parser().parse_args())
