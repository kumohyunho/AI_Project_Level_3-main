"""MOTChallenge 형식 시퀀스 읽기와 검출 캐시.

시퀀스 폴더 구조: ``<seq>/seqinfo.ini``, ``<seq>/img1/000001.jpg ...``, ``<seq>/gt/gt.txt``.
``prepare_dataset.py mot``가 만든 train/val/test 구간 폴더도 같은 구조입니다.

검출은 추적기 설정과 무관하므로, 한 번만 실행해 ``results/cache``에 저장하고
여러 추적기 설정(기본·개선·튜닝 후보)을 같은 검출 위에서 비교합니다.
"""
import configparser
from pathlib import Path
from time import perf_counter

import numpy as np

from pipeline import ROOT, Detector, TRACKABLE_CLASSES
from tracker import extract_features

CACHE_DIR = ROOT / "results" / "cache"
CACHE_CONF = 0.05      # 튜닝 범위의 최솟값보다 낮게 저장합니다.
FEATURE_CONF = 0.3     # 이 점수 이상인 검출만 외형 특징을 계산합니다.


class Sequence:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        parser = configparser.ConfigParser()
        with (self.folder / "seqinfo.ini").open(encoding="utf-8-sig") as source:
            parser.read_file(source)
        info = parser["Sequence"]
        self.name = info.get("name", self.folder.name)
        self.length = info.getint("seqLength")
        self.fps = info.getfloat("frameRate", fallback=30.0)
        self.width = info.getint("imWidth", fallback=0)
        self.height = info.getint("imHeight", fallback=0)
        self.image_dir = self.folder / info.get("imDir", "img1")
        self.extension = info.get("imExt", ".jpg")
        self.gt_path = self.folder / "gt" / "gt.txt"

    def frame_path(self, number):
        return self.image_dir / f"{number:06d}{self.extension}"

    def read(self, number):
        import cv2

        path = self.frame_path(number)
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"프레임을 읽을 수 없습니다: {path}")
        return frame

    @property
    def frames(self):
        return list(range(1, self.length + 1))


def find_sequences(paths):
    """폴더 목록에서 ``seqinfo.ini``가 있는 시퀀스를 찾습니다 (상위 폴더를 주면 하위를 모두 사용)."""
    found = []
    for path in map(Path, paths):
        if (path / "seqinfo.ini").is_file():
            found.append(Sequence(path))
        else:
            found += [Sequence(p.parent) for p in sorted(path.glob("*/seqinfo.ini"))]
    if not found:
        raise FileNotFoundError(f"seqinfo.ini가 있는 시퀀스를 찾지 못했습니다: {paths}")
    return found


def cache_path(weights, sequence, imgsz):
    """train/val/test 구간 폴더의 시퀀스 이름이 같으므로 폴더 경로 해시를 키에 넣습니다."""
    import hashlib

    digest = hashlib.md5(str(sequence.folder).encode("utf-8")).hexdigest()[:8]
    split = sequence.folder.parent.name
    return CACHE_DIR / f"{Path(weights).stem}_{imgsz}" / f"{split}_{sequence.name}_{digest}.npz"


def load_detections(sequence, weights, imgsz=640, device="cpu", refresh=False):
    """시퀀스 전체 검출(+외형 특징)을 캐시에서 읽거나, 없으면 실행해 저장합니다."""
    path = cache_path(weights, sequence, imgsz)
    if path.is_file() and not refresh:
        data = np.load(path, allow_pickle=False)
        return _unpack(data)

    detector = Detector(weights, ["person"], imgsz, device)
    frames, boxes, scores, features, det_ms = [], [], [], [], []
    for number in sequence.frames:
        image = sequence.read(number)
        started = perf_counter()
        b, s, _ = detector(image, CACHE_CONF)
        det_ms.append((perf_counter() - started) * 1000)
        strong = s >= FEATURE_CONF
        vectors = np.full((len(b), 128), np.nan, dtype=np.float32)
        for index, vector in zip(np.flatnonzero(strong), extract_features(image, b[strong])):
            if vector is not None:
                vectors[index] = vector
        frames.append(np.full(len(b), number))
        boxes.append(b)
        scores.append(s)
        features.append(vectors)
        print(f"\r{sequence.name}: {number}/{sequence.length}", end="", flush=True)
    print()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=np.concatenate(frames), boxes=np.concatenate(boxes).reshape(-1, 4),
                        scores=np.concatenate(scores), features=np.concatenate(features).reshape(-1, 128),
                        det_ms=np.array(det_ms))
    return _unpack(np.load(path, allow_pickle=False))


def _unpack(data):
    frames = data["frames"]
    per_frame = {}
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    boxes, scores, features = data["boxes"][order], data["scores"][order], data["features"][order]
    starts = np.searchsorted(frames, np.unique(frames))
    ends = np.r_[starts[1:], len(frames)]
    for start, end in zip(starts, ends):
        vectors = [None if np.isnan(v[0]) else v for v in features[start:end]]
        per_frame[int(frames[start])] = (boxes[start:end], scores[start:end], vectors)
    return per_frame, float(np.mean(data["det_ms"])) if len(data["det_ms"]) else 0.0


__all__ = ["Sequence", "find_sequences", "load_detections", "TRACKABLE_CLASSES"]
