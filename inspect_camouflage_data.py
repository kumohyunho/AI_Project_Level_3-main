"""전처리한 보호색 데이터의 구조 검사와 정답 박스 미리보기. 모델 학습·추론은 하지 않습니다.

실행: python inspect_camouflage_data.py datasets/camouflage --check-hashes
자세한 판정 범위: docs/CAMOUFLAGE_DATA_CHECK.md.
"""
import argparse
import configparser
import hashlib
import html
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw
import yaml

from extensions.camouflage import CLASS_NAME

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_yolo(path):
    """박스는 0 기반 정규화 cx,cy,w,h. 화면 밖·NaN·다른 클래스는 허용하지 않습니다."""
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"{number}행: class=0인 YOLO 5열 형식이어야 합니다.")
        cx, cy, width, height = map(float, fields[1:])
        values = (cx, cy, width, height)
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or min(width, height) <= 0:
            raise ValueError(f"{number}행: 유한한 0~1 좌표와 양의 크기가 필요합니다.")
        box = (cx-width/2, cy-height/2, cx+width/2, cy+height/2)
        if min(box) < -1e-7 or max(box) > 1 + 1e-7:
            raise ValueError(f"{number}행: 이미지 경계를 벗어난 박스입니다.")
        boxes.append(box)
    return boxes


def select_samples(records, limit):
    """분할·출처·영상별로 번갈아 고르는 결정적 표본. 전체 데이터 검증을 대신하지 않습니다."""
    groups = defaultdict(list)
    for row in sorted(records, key=lambda r: r["image"]):
        groups[(row["split"], row["source"], row.get("sequence") or "")].append(row)
    queues = deque(deque(groups[key]) for key in sorted(groups))
    selected = []
    while queues and len(selected) < limit:
        group = queues.popleft()
        selected.append(group.popleft())
        if group:
            queues.append(group)
    return selected


class Inspector:
    def __init__(self, root, check_hashes):
        self.root = root
        self.check_hashes = check_hashes
        self.issues, self.records, self.sequences = [], [], []
        self.detections, self.mot_previews = {}, {}
        self.hash_splits = {}

    def issue(self, code, path, message, warning=False):
        self.issues.append(dict(severity="warning" if warning else "error", code=code,
                                path=str(path), message=str(message)))

    def within(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("데이터 폴더 밖을 가리키는 경로입니다.")
        return path

    def checked(self, code, path, action):
        try:
            return action()
        except (OSError, ValueError, TypeError, KeyError, AttributeError, configparser.Error, yaml.YAMLError) as error:
            self.issue(code, path, error)
            return None

    def audit(self):
        manifest_path = self.root / "splits.json"
        manifest = self.checked("manifest", manifest_path,
                                lambda: json.loads(manifest_path.read_text(encoding="utf-8")))
        if not isinstance(manifest, dict):
            if manifest is not None:
                self.issue("manifest", manifest_path, "JSON 객체여야 합니다.")
            return
        if manifest.get("classes") != [CLASS_NAME]:
            self.issue("classes", manifest_path, "보호색 동물 단일 클래스 데이터가 아닙니다.")
        self.checked("yaml", self.root / "dataset.yaml", self.audit_yaml)
        records = manifest.get("records")
        if not isinstance(records, list) or not records:
            self.issue("manifest", manifest_path, "records는 비어 있지 않은 목록이어야 합니다.")
            return
        seen_paths, sequence_splits = set(), {}
        for row in records:
            self.checked("record", manifest_path,
                         lambda row=row: self.audit_record(row, seen_paths, self.hash_splits, sequence_splits))
        for split in SPLITS:
            rows = [r for r in self.records if r["split"] == split]
            actual = dict(images=len(rows), boxes=sum(len(self.detections.get(r["image"], {}).get("boxes", [])) for r in rows))
            if not rows:
                self.issue("empty_split", split, "분할에 이미지가 없습니다.")
            if manifest.get("counts", {}).get(split) != actual:
                self.issue("counts", split, f"splits.json의 개수와 검사 결과가 다릅니다: {actual}")
            self.audit_extra_files(split)
        declared = manifest.get("sequences", [])
        if not isinstance(declared, list):
            self.issue("sequences", manifest_path, "sequences는 목록이어야 합니다.")
            declared = []
        declared_paths = set()
        for seq in declared:
            self.checked("sequence", manifest_path,
                         lambda seq=seq: self.audit_sequence(seq, declared_paths))
        for path in (self.root / "tracking").glob("*/*/seqinfo.ini"):
            if path.parent.resolve() not in declared_paths:
                self.issue("unlisted_sequence", path, "데이터 명세에 등록되지 않은 추적 시퀀스입니다.")
        registered = {(s.get("split"), s.get("sequence")) for s in declared if isinstance(s, dict)}
        for row in self.records:
            if row["source"] == "moca" and (row["split"], row.get("sequence")) not in registered:
                self.issue("missing_sequence", row["image"], "MoCA 이미지에 대응하는 추적 시퀀스가 없습니다.")

    def audit_yaml(self):
        data = yaml.safe_load((self.root / "dataset.yaml").read_text(encoding="utf-8"))
        if data.get("names") not in ({0: CLASS_NAME}, [CLASS_NAME]):
            raise ValueError("dataset.yaml의 클래스가 보호색 동물 단일 클래스와 다릅니다.")
        for split in SPLITS:
            if data.get(split) != f"images/{split}":
                raise ValueError(f"{split}은 images/{split}이어야 합니다.")
        configured = Path(data["path"])
        if not configured.is_absolute():
            configured = self.root / configured
        if configured.resolve() != self.root:
            self.issue("yaml_path", "dataset.yaml", "데이터를 옮겼다면 path를 현재 위치에 맞추세요.", warning=True)

    def audit_record(self, row, seen_paths, hash_splits, sequence_splits):
        relative, split, source = row["image"], row["split"], row["source"]
        parts = Path(relative).parts
        if split not in SPLITS or len(parts) != 3 or parts[:2] != ("images", split):
            raise ValueError("image 경로는 images/<split>/<파일>이어야 합니다.")
        if source not in ("cod10k", "moca") or Path(relative).suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("출처 또는 이미지 확장자가 올바르지 않습니다.")
        if relative in seen_paths:
            raise ValueError(f"중복 이미지 레코드입니다: {relative}")
        seen_paths.add(relative)
        image = self.within(relative)
        self.records.append(row)
        if source == "moca":
            name = row.get("sequence")
            if not isinstance(name, str) or not name:
                raise ValueError("MoCA 레코드에는 영상 이름이 필요합니다.")
            if sequence_splits.setdefault(name, split) != split:
                self.issue("sequence_leak", relative, "같은 영상이 서로 다른 분할에 있습니다.")
        digest = row.get("sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("SHA256 형식이 올바르지 않습니다.")
        if self.check_hashes:
            actual = hashlib.sha256(image.read_bytes()).hexdigest()
            if actual != digest:
                self.issue("hash_changed", relative, "현재 파일 해시가 전처리 명세와 다릅니다.")
            digest = actual
        if hash_splits.setdefault(digest, split) != split:
            self.issue("image_leak", relative, "동일 이미지 해시가 서로 다른 분할에 있습니다.")
        with Image.open(image) as original:
            original.load()
            size = original.size
        label = self.within(f"labels/{split}/{image.stem}.txt")
        boxes = read_yolo(label)
        if type(row["boxes"]) is not int or row["boxes"] != len(boxes):
            self.issue("box_count", label, "레코드의 박스 개수와 YOLO 라벨이 다릅니다.")
        self.detections[relative] = dict(boxes=boxes, size=size)

    def audit_extra_files(self, split):
        expected_images = {r["image"] for r in self.records}
        expected_labels = {f"labels/{r['split']}/{Path(r['image']).stem}.txt" for r in self.records}
        for folder, expected in (("images", expected_images), ("labels", expected_labels)):
            for path in (self.root / folder / split).rglob("*"):
                if path.is_file() and str(path.relative_to(self.root)) not in expected:
                    self.issue("unlisted_file", path.relative_to(self.root), "명세에 없는 파일입니다. 용도를 확인하세요.", warning=True)

    def audit_sequence(self, summary, declared_paths):
        name, split = summary["sequence"], summary["split"]
        if split not in SPLITS or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("추적 시퀀스 이름/분할이 올바르지 않습니다.")
        folder = self.within(f"tracking/{split}/{name}")
        if folder in declared_paths:
            raise ValueError("추적 시퀀스가 중복 등록되어 있습니다.")
        declared_paths.add(folder)
        info = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
        parser = configparser.ConfigParser()
        with (folder / "seqinfo.ini").open(encoding="utf-8-sig") as handle:
            parser.read_file(handle)
        seq = parser["Sequence"]
        length, width, height = (seq.getint(k) for k in ("seqLength", "imWidth", "imHeight"))
        fps = seq.getfloat("frameRate")
        if min(length, width, height) <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("프레임 수·크기·FPS가 올바르지 않습니다.")
        evaluated = info["evaluated_frames"]
        if (not isinstance(evaluated, list) or not evaluated
                or any(type(v) is not int or not 1 <= v <= length for v in evaluated)
                or evaluated != sorted(set(evaluated))):
            raise ValueError("평가 프레임은 범위 안의 중복 없는 정수 목록이어야 합니다.")
        for key, expected in (("sequence", name), ("split", split), ("frames", length),
                              ("evaluated_frames", evaluated), ("ids_verified", info["ids_verified"])):
            if info[key] != expected or summary[key] != expected:
                raise ValueError(f"추적 명세와 시퀀스 정보가 다릅니다: {key}")
        if type(info["ids_verified"]) is not bool:
            raise ValueError("ID 확인 상태는 true/false여야 합니다.")
        extension = seq["imExt"]
        if extension not in IMAGE_SUFFIXES or seq["imDir"] != "img1" or seq["name"] != name:
            raise ValueError("전처리 시퀀스의 img1/이미지 확장자를 확인하세요.")
        frame_paths = []
        for number in range(1, length+1):
            path = self.within(str((folder / "img1" / f"{number:06d}{extension}").relative_to(self.root)))
            with Image.open(path) as frame:
                frame.load()
                if frame.size != (width, height):
                    raise ValueError(f"프레임 해상도가 다릅니다: {path.name}")
            if str(number) not in info["original_files"]:
                raise ValueError(f"원본 프레임 매핑이 없습니다: {number}")
            frame_paths.append(path)
            if self.check_hashes:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if self.hash_splits.setdefault(digest, split) != split:
                    self.issue("tracking_image_leak", path.relative_to(self.root), "동일 추적 프레임이 서로 다른 분할에 있습니다.")
        expected_paths = set(frame_paths)
        for extra in (folder / "img1").iterdir():
            if extra.is_file() and extra not in expected_paths:
                self.issue("unlisted_frame", extra.relative_to(self.root), "시퀀스 길이에 포함되지 않은 파일입니다.", warning=True)
        gt_path = folder / "gt" / "gt.txt"
        ground_truth = defaultdict(list)
        if info["ids_verified"]:
            seen_ids = set()
            for line in gt_path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                values = list(map(float, line.split(",")))
                if len(values) != 9 or not all(math.isfinite(v) for v in values):
                    raise ValueError("MOT 정답은 유한한 값 9열이어야 합니다.")
                frame, pid, left, top, w, h, flag, cls, visibility = values
                if frame != int(frame) or pid != int(pid) or pid < 1 or frame not in evaluated:
                    raise ValueError("MOT 프레임/ID가 평가 명세와 맞지 않습니다.")
                box = (left-1, top-1, left-1+w, top-1+h)
                if (min(w, h) <= 0 or min(box) < -1e-4 or box[2] > width+1e-4 or box[3] > height+1e-4
                        or flag != 1 or cls != 1 or not 0 <= visibility <= 1):
                    raise ValueError("MOT 박스 경계·플래그·가시 비율을 확인하세요.")
                if (frame, pid) in seen_ids:
                    raise ValueError("한 프레임에 같은 ID가 중복되어 있습니다.")
                seen_ids.add((frame, pid))
                ground_truth[int(frame)].append((int(pid), box))
        else:
            self.issue("ids_unverified", gt_path.relative_to(self.root), "ID 정답이 선언되지 않아 추적 평가를 할 수 없습니다.", warning=True)
            if gt_path.exists():
                self.issue("unexpected_gt", gt_path, "ID 미확인 시퀀스에 gt.txt가 있습니다.")
        found = set()
        for row in self.records:
            if row["source"] != "moca" or row.get("sequence") != name or row["split"] != split:
                continue
            number = int(Path(row["image"]).stem.rsplit("_", 1)[-1])
            found.add(number)
            if number not in evaluated:
                raise ValueError("미주석 프레임이 검출 학습 이미지에 포함되어 있습니다.")
            detection = self.detections.get(row["image"])
            if detection is None:
                continue
            expected_pixels = sorted((x1*width, y1*height, x2*width, y2*height) for x1, y1, x2, y2 in detection["boxes"])
            if info["ids_verified"]:
                actual_pixels = sorted(box for _, box in ground_truth[number])
                if len(expected_pixels) != len(actual_pixels) or any(
                        abs(a-b) > 0.001 for a_box, b_box in zip(expected_pixels, actual_pixels) for a, b in zip(a_box, b_box)):
                    self.issue("gt_label_mismatch", row["image"], "MOT 정답 박스와 YOLO 라벨이 다릅니다.")
                self.mot_previews[row["image"]] = ground_truth[number]
            with Image.open(self.within(row["image"])) as detection_image, Image.open(frame_paths[number-1]) as tracking_image:
                if (detection_image.size != tracking_image.size
                        or detection_image.convert("RGB").tobytes() != tracking_image.convert("RGB").tobytes()):
                    self.issue("frame_mismatch", row["image"], "검출 학습 이미지와 추적 프레임의 내용이 다릅니다.")
        if found != set(evaluated):
            self.issue("annotation_coverage", folder, "평가 프레임과 검출 라벨 프레임 목록이 다릅니다.")
        self.sequences.append(dict(sequence=name, split=split, frames=length, fps=fps,
                                   annotated_frames=len(evaluated), ids_declared=info["ids_verified"]))

    def previews(self, output, limit):
        rows = [r for r in self.records if r["image"] in self.detections]
        previews = []
        for index, row in enumerate(select_samples(rows, limit)):
            relative = row["image"]
            with Image.open(self.within(relative)) as source:
                image = source.convert("RGB")
            width, height = image.size
            draw = ImageDraw.Draw(image)
            for x1, y1, x2, y2 in self.detections[relative]["boxes"]:
                pixels = (max(0, min(width-1, x1*width)), max(0, min(height-1, y1*height)),
                          max(0, min(width-1, x2*width)), max(0, min(height-1, y2*height)))
                draw.rectangle(pixels, outline="#00ff70", width=2)
            for pid, box in self.mot_previews.get(relative, []):
                x1, y1, x2, y2 = box
                draw.text((max(0, x1), max(0, y1-12)), f"ID {pid}", fill="#00ffff", stroke_width=1, stroke_fill="black")
            image.thumbnail((960, 640))
            target = f"preview_{index:03d}.png"
            image.save(output / target)
            previews.append(dict(image=relative, preview=target, split=row["split"], source=row["source"],
                                 sequence=row.get("sequence"), declared_ids=[pid for pid, _ in self.mot_previews.get(relative, [])]))
        return previews


def inspect(dataset, output, samples=12, check_hashes=False):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {dataset}")
    if samples < 0:
        raise ValueError("미리보기 개수는 0 이상이어야 합니다.")
    if output.exists() or output.is_relative_to(dataset):
        raise ValueError("기존 폴더나 데이터 내부에 결과를 쓰지 않습니다. 새로운 외부 경로를 지정하세요.")
    audit = Inspector(dataset, check_hashes)
    audit.checked("dataset", dataset, audit.audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as folder:
        staging = Path(folder) / "report"
        staging.mkdir()
        previews = audit.previews(staging, samples)
        errors = sum(i["severity"] == "error" for i in audit.issues)
        report = dict(dataset=str(dataset), errors=errors, warnings=len(audit.issues)-errors,
                      status="구조 오류 있음" if errors else "구조 검사 통과·사람의 시각 검토 필요",
                      hashes_recomputed=check_hashes, inspected_records=len(audit.records),
                      sequences=audit.sequences, issues=audit.issues, previews=previews,
                      limits=["실제 보호색 여부·종·개체 ID의 의미·정답의 완전성은 자동 확인하지 않음",
                              "학습·검출·추적 성능 측정 미실행", "미리보기는 표본이며 전체 영상 검토를 대신하지 않음",
                              "원본 주석과의 일치·유사 이미지·동일 원본 영상 누수는 별도 확인 필요",
                              "실제 이미지 해시 재계산 수행" if check_hashes else "해시는 명세 값만 비교. 실제 파일 검증에는 --check-hashes 필요"])
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        issue_html = "".join(f"<li>{html.escape(i['severity'])}: {html.escape(i['code'])} — "
                             f"{html.escape(i['path'])}: {html.escape(i['message'])}</li>" for i in audit.issues)
        cards = "".join(f"<figure><img src='{p['preview']}' alt='정답 박스 미리보기'>"
                        f"<figcaption>{html.escape(p['image'])}<br>선언된 ID: {html.escape(str(p['declared_ids']))}"
                        "</figcaption></figure>" for p in previews)
        page = ("<!doctype html><html lang='ko'><meta charset='utf-8'><title>보호색 데이터 검사</title>"
                "<style>body{font-family:sans-serif;max-width:1100px;margin:32px auto;padding:16px}"
                "img{max-width:100%;height:auto}figure{margin:24px 0}li{margin:8px 0}</style>"
                f"<h1>보호색 데이터 검사</h1><p>{html.escape(report['status'])}</p>"
                f"<p>오류 {errors}개 / 경고 {report['warnings']}개 / 검사 레코드 {len(audit.records)}개</p>"
                "<p>녹색: YOLO 정답 박스 / 청록색 ID: MOT에 기록된 ID. 모델 예측 결과가 아닙니다.</p>"
                "<h2>검사 범위의 한계</h2><ul>" + "".join(f"<li>{html.escape(v)}</li>" for v in report["limits"])
                + f"</ul><h2>발견 사항</h2><ul>{issue_html or '<li>구조 오류·경고 없음</li>'}</ul>"
                + f"<h2>사람이 확인할 표본</h2>{cards}</html>")
        (staging / "index.html").write_text(page, encoding="utf-8")
        staging.rename(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/camouflage/data-check"))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--check-hashes", action="store_true", help="실제 파일 SHA256 재계산 (시간이 더 걸림)")
    args = parser.parse_args(argv)
    report = inspect(args.dataset, args.output, args.samples, args.check_hashes)
    print(f"{report['status']}: 오류 {report['errors']}, 경고 {report['warnings']} → {args.output / 'index.html'}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
