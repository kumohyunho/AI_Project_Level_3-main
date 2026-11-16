"""4개 공개 데이터셋을 학습·검증·테스트로 나누고 모델 입력 형식으로 전처리합니다 (요구 사항 1).

| 종류  | 원본                  | 결과                                   | train / val / test                      |
|-------|-----------------------|----------------------------------------|-----------------------------------------|
| coco  | COCO 2017 instances   | YOLO 검출 (person·cat·dog·bird 4클래스) | train2017을 나눔 / 〃 / val2017         |
| crowd | CrowdHuman .odgt      | YOLO 검출 (person, full box)           | train을 나눔 / 〃 / 공식 val            |
| face  | WIDER FACE bbx_gt.txt | YOLO 검출 (face)                       | train을 나눔 / 〃 / 공식 val            |
| mot   | MOT17·MOT20 train     | YOLO 검출 + **ID가 있는 추적용 시퀀스** | 시퀀스마다 앞 50% / 다음 25% / 마지막 25% |

공식 test 세트는 정답이 공개되지 않아, 공식 val(또는 MOT train 뒷부분)을 테스트로 사용하고
train의 일부를 검증용으로 떼어 냅니다. 이미지는 하드링크(불가하면 복사)로 연결해 용량을 아낍니다.

MOT 추적용 시퀀스(``tracking/<split>/<seq>``)는 프레임 번호를 1부터 다시 매기고, 정답의 객체 ID·위치·
가시 비율을 그대로 보존한 ``gt/gt.txt``와 ``seqinfo.ini``를 만듭니다. 평가·튜닝·실패 분석이 이 폴더를 입력으로 씁니다.

예)
    python prepare_dataset.py mot --sequences datasets/raw/MOT17/train datasets/raw/MOT20/train --output datasets/mot
    python prepare_dataset.py coco --train-images datasets/raw/coco/train2017 --train-labels datasets/raw/coco/annotations/instances_train2017.json \\
        --val-images datasets/raw/coco/val2017 --val-labels datasets/raw/coco/annotations/instances_val2017.json --output datasets/coco4
"""
import argparse
import configparser
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

COCO_CLASSES = {1: 0, 18: 1, 17: 2, 16: 3}   # COCO category_id → 0 person, 1 dog, 2 cat, 3 bird
COCO_NAMES = ["person", "dog", "cat", "bird"]


# ----------------------------------------------------------------- 원본 라벨 읽기
def coco_records(annotation):
    data = json.loads(Path(annotation).read_text(encoding="utf-8"))
    images = {image["id"]: image["file_name"] for image in data["images"]}
    boxes = defaultdict(list)
    for item in data["annotations"]:
        if item["category_id"] in COCO_CLASSES and not item.get("iscrowd"):
            boxes[item["image_id"]].append((COCO_CLASSES[item["category_id"]], *item["bbox"]))
    for image_id, items in sorted(boxes.items()):
        yield Path(images[image_id]), items


def crowd_records(annotation):
    with Path(annotation).open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            boxes = [(0, *entry["fbox"]) for entry in record["gtboxes"]
                     if entry["tag"] == "person" and not entry.get("extra", {}).get("ignore")]
            yield Path(record["ID"] + ".jpg"), boxes


def wider_records(annotation):
    lines = Path(annotation).read_text(encoding="utf-8-sig").splitlines()
    index = 0
    while index < len(lines):
        name = lines[index].strip()
        index += 1
        if not name:
            continue
        count = int(lines[index].strip())
        index += 1
        boxes = []
        for _ in range(count):
            fields = [int(value) for value in lines[index].split()]
            index += 1
            if len(fields) < 10:
                raise ValueError(f"잘못된 WIDER FACE 박스: {name}")
            if fields[7] == 0:  # invalid 플래그
                boxes.append((0, *fields[:4]))
        if count == 0 and index < len(lines):
            dummy = lines[index].split()
            if len(dummy) == 10 and all(value == "0" for value in dummy):
                index += 1
        yield Path(name), boxes


# ----------------------------------------------------------------- 공통 변환
def normalize(box, width, height):
    cls, x, y, w, h = box
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + w), min(height, y + h)
    if right <= left or bottom <= top:
        return None
    return (int(cls), (left + right) / (2 * width), (top + bottom) / (2 * height),
            (right - left) / width, (bottom - top) / height)


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def in_validation(name, ratio):
    """파일 이름 해시로 나눠 실행할 때마다 같은 분할이 나오게 합니다."""
    digest = int(hashlib.md5(str(name).encode("utf-8")).hexdigest()[:8], 16)
    return digest / 0xFFFFFFFF < ratio


def convert_split(records, source, destination, split, limit=None):
    from PIL import Image

    images = destination / "images" / split
    labels = destination / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    image_count, box_count = 0, Counter()
    for name, boxes in records:
        if limit and image_count >= limit:
            break
        if name.is_absolute() or ".." in name.parts or name.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"잘못된 이미지 경로: {name}")
        image_path, output = Path(source) / name, images / name
        if output.exists():
            raise ValueError(f"중복 이미지: {name}")
        with Image.open(image_path) as image:
            width, height = image.size
        link(image_path, output)
        label = (labels / name).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        values = [v for v in (normalize(box, width, height) for box in boxes) if v is not None]
        label.write_text("".join(f"{v[0]} " + " ".join(f"{x:.8f}" for x in v[1:]) + "\n" for v in values),
                         encoding="utf-8")
        image_count += 1
        box_count.update(v[0] for v in values)
    if not image_count or not box_count:
        raise ValueError(f"{split}: 이미지 또는 유효한 정답 박스가 없습니다.")
    print(f"{split}: {image_count} images, {sum(box_count.values())} boxes")
    return dict(images=image_count, boxes=dict(box_count))


def write_yaml(staging, output, names):
    listing = ", ".join(f"{i}: {n}" for i, n in enumerate(names))
    (staging / "dataset.yaml").write_text(
        f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
        f"train: images/train\nval: images/val\ntest: images/test\nnames: {{{listing}}}\n", encoding="utf-8")


def prepare_images(args):
    reader = {"coco": coco_records, "crowd": crowd_records, "face": wider_records}[args.kind]
    names = {"coco": COCO_NAMES, "crowd": ["person"], "face": ["face"]}[args.kind]
    with staged(args.output) as (staging, output):
        train_records = list(reader(args.train_labels))
        summary = {
            "train": convert_split((r for r in train_records if not in_validation(r[0], args.val_ratio)),
                                   args.train_images, staging, "train", args.limit),
            "val": convert_split((r for r in train_records if in_validation(r[0], args.val_ratio)),
                                 args.train_images, staging, "val", args.limit),
            "test": convert_split(reader(args.val_labels), args.val_images, staging, "test", args.limit),
        }
        write_yaml(staging, output, names)
        (staging / "splits.json").write_text(json.dumps(dict(
            kind=args.kind, names=names, val_ratio=args.val_ratio, counts=summary,
            note="test = 공식 val, val = 공식 train에서 파일명 해시로 분리"), ensure_ascii=False, indent=2),
            encoding="utf-8")


# ----------------------------------------------------------------- MOT17 / MOT20
def mot_sequences(paths, detector):
    """MOT17은 같은 영상이 DPM/FRCNN/SDP로 3번 들어 있어 하나만 고릅니다."""
    found = []
    for path in map(Path, paths):
        candidates = [path] if (path / "seqinfo.ini").is_file() else sorted(p.parent for p in path.glob("*/seqinfo.ini"))
        for sequence in candidates:
            name = sequence.name
            if name.startswith("MOT17") and not name.endswith(f"-{detector}"):
                continue
            if not (sequence / "gt" / "gt.txt").is_file():
                raise FileNotFoundError(f"정답이 없는 시퀀스입니다 (test 세트는 사용할 수 없음): {sequence}")
            found.append(sequence)
    if not found:
        raise FileNotFoundError("MOT 시퀀스를 찾지 못했습니다.")
    return found


def read_seqinfo(sequence):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    with (sequence / "seqinfo.ini").open(encoding="utf-8-sig") as source:
        parser.read_file(source)
    return parser


def prepare_mot(args):
    ratios = (args.train_ratio, args.val_ratio, 1 - args.train_ratio - args.val_ratio)
    if min(ratios) <= 0:
        raise ValueError("train/val 비율의 합은 1보다 작아야 합니다.")
    with staged(args.output) as (staging, output):
        summary = defaultdict(dict)
        for sequence in mot_sequences(args.sequences, args.detector):
            info = read_seqinfo(sequence)
            seq = info["Sequence"]
            length, extension = int(seq["seqLength"]), seq.get("imExt", ".jpg")
            width, height = int(seq["imWidth"]), int(seq["imHeight"])
            first_val = int(length * ratios[0]) + 1
            first_test = int(length * (ratios[0] + ratios[1])) + 1
            ranges = {"train": range(1, first_val), "val": range(first_val, first_test),
                      "test": range(first_test, length + 1)}
            gt = defaultdict(list)
            with (sequence / "gt" / "gt.txt").open(encoding="utf-8-sig") as source:
                for line in source:
                    if line.strip():
                        values = line.strip().split(",")
                        gt[int(values[0])].append(values)
            for split, frames in ranges.items():
                # (1) 검출 학습용 YOLO 라벨: 보행자(class 1, flag 1) 중 30% 이상 보이는 박스
                det_images = staging / "images" / split
                det_labels = staging / "labels" / split
                det_images.mkdir(parents=True, exist_ok=True)
                det_labels.mkdir(parents=True, exist_ok=True)
                for number in frames[::args.stride]:
                    name = f"{sequence.name}_{number:06d}"
                    link(sequence / seq.get("imDir", "img1") / f"{number:06d}{extension}",
                         det_images / f"{name}{extension}")
                    boxes = [(0, *map(float, v[2:6])) for v in gt[number]
                             if int(v[6]) == 1 and int(v[7]) == 1 and float(v[8]) >= 0.3]
                    rows = [normalize(b, width, height) for b in boxes]
                    (det_labels / f"{name}.txt").write_text(
                        "".join("0 " + " ".join(f"{x:.8f}" for x in r[1:]) + "\n" for r in rows if r),
                        encoding="utf-8")
                # (2) 추적 평가용 시퀀스: ID·위치·가시 비율 그대로, 프레임 번호만 1부터
                target = staging / "tracking" / split / sequence.name
                lines = []
                for new_number, number in enumerate(frames, start=1):
                    link(sequence / seq.get("imDir", "img1") / f"{number:06d}{extension}",
                         target / "img1" / f"{new_number:06d}{extension}")
                    lines += [",".join([str(new_number)] + v[1:]) + "\n" for v in gt[number]]
                (target / "gt").mkdir(parents=True, exist_ok=True)
                (target / "gt" / "gt.txt").write_text("".join(lines), encoding="utf-8")
                new_info = configparser.ConfigParser()
                new_info.optionxform = str
                new_info["Sequence"] = dict(seq, name=sequence.name, seqLength=str(len(frames)), imDir="img1")
                with (target / "seqinfo.ini").open("w", encoding="utf-8") as handle:
                    new_info.write(handle, space_around_delimiters=False)
                ids = {v[1] for n in frames for v in gt[n] if int(v[6]) == 1 and int(v[7]) == 1}
                summary[split][sequence.name] = dict(frames=[frames.start, frames.stop - 1], ids=len(ids))
            print(f"{sequence.name}: train {len(ranges['train'])} / val {len(ranges['val'])} / test {len(ranges['test'])} 프레임")
        write_yaml(staging, output, ["person"])
        (staging / "splits.json").write_text(json.dumps(dict(
            kind="mot", ratios=ratios, detection_stride=args.stride, detector_variant=args.detector,
            splits=summary), ensure_ascii=False, indent=2), encoding="utf-8")


class staged:
    """임시 폴더에 만든 뒤 성공했을 때만 최종 경로로 옮깁니다."""

    def __init__(self, output):
        self.output = Path(output).resolve()

    def __enter__(self):
        if self.output.exists():
            raise FileExistsError(f"출력 경로가 이미 있습니다: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=self.output.parent)
        self.staging = Path(self.temporary.name) / "dataset"
        self.staging.mkdir()
        return self.staging, self.output

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.staging.rename(self.output)
                print(self.output / "dataset.yaml")
        finally:
            self.temporary.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)
    for kind in ("coco", "crowd", "face"):
        p = sub.add_parser(kind)
        for split in ("train", "val"):
            p.add_argument(f"--{split}-images", type=Path, required=True)
            p.add_argument(f"--{split}-labels", type=Path, required=True)
        p.add_argument("--val-ratio", type=float, default=0.1, help="train에서 떼어 낼 검증 비율")
        p.add_argument("--limit", type=int, help="분할마다 최대 이미지 수 (빠른 실험용)")
        p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("mot")
    p.add_argument("--sequences", nargs="+", required=True, help="MOT17/train, MOT20/train 또는 시퀀스 폴더")
    p.add_argument("--detector", default="FRCNN", help="MOT17에서 사용할 버전 (DPM/FRCNN/SDP)")
    p.add_argument("--train-ratio", type=float, default=0.5)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=5, help="검출 학습 이미지 간격 (연속 프레임 중복 줄이기)")
    p.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    (prepare_mot if arguments.kind == "mot" else prepare_images)(arguments)
