"""검증 분할에서 보호색 검출 신뢰도·NMS IoU를 비교합니다. 추적 ID 평가는 아닙니다."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from extensions.camouflage import CLASS_NAME


def overlap(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) -
                              np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    union = np.prod(box[2:] - box[:2]) + (boxes[:, 2:] - boxes[:, :2]).prod(axis=1) - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_counts(truth, predictions, confidence):
    """점수 내림차순, 정답과 일대일 IoU>=0.5 매칭. 중복 후보는 FP의 부분집합입니다."""
    truth = np.asarray(truth, dtype=float).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=float).reshape(-1, 5)
    used = set()
    tp = fp = duplicates = 0
    for prediction in sorted(predictions, key=lambda row: -row[4]):
        if prediction[4] < confidence:
            continue
        ious = overlap(prediction[:4], truth)
        available = [i for i in range(len(truth)) if i not in used and ious[i] >= 0.5]
        if available:
            used.add(max(available, key=lambda i: ious[i]))
            tp += 1
        else:
            fp += 1
            duplicates += int(bool(np.any(ious >= 0.5)))
    return dict(tp=tp, fp=fp, fn=len(truth) - tp, duplicate_candidates=duplicates)


def metrics(counts):
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return dict(counts, precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def validation_images(root):
    """명세에 있는 val 전체만 읽고 라벨·경로를 검사합니다. test 선택 옵션은 없습니다."""
    root = Path(root).resolve()
    manifest = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    if manifest.get("classes") != [CLASS_NAME]:
        raise ValueError("보호색 단일 클래스 데이터가 필요합니다.")
    records = [row for row in manifest["records"] if row["split"] == "val"]
    if not records or len(records) != manifest["counts"]["val"]["images"]:
        raise ValueError("val 이미지 개수가 명세와 다르거나 비어 있습니다.")
    prepared, seen = [], set()
    for row in records:
        relative = Path(row["image"])
        if relative.is_absolute() or len(relative.parts) != 3 or relative.parts[:2] != ("images", "val"):
            raise ValueError("val 이미지 경로를 확인하세요.")
        image = (root / relative).resolve()
        label = (root / "labels" / "val" / relative.with_suffix(".txt").name).resolve()
        if not image.is_relative_to(root) or not label.is_relative_to(root) or image in seen:
            raise ValueError("데이터 경로 이탈 또는 중복 이미지입니다.")
        seen.add(image)
        with Image.open(image) as handle:
            width, height = handle.size
        boxes = []
        for line in label.read_text(encoding="utf-8").splitlines():
            values = list(map(float, line.split()))
            if len(values) != 5 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"잘못된 YOLO 라벨: {label}")
            cls, x, y, w, h = values
            if cls != 0 or not 0 < w <= 1 or not 0 < h <= 1 or not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError(f"라벨 클래스·좌표 범위 오류: {label}")
            box = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
            if min(box) < -0.001 or box[2] > width+0.001 or box[3] > height+0.001:
                raise ValueError(f"영상 밖 정답 박스: {label}")
            boxes.append(box)
        if len(boxes) != row["boxes"]:
            raise ValueError(f"정답 개수가 명세와 다릅니다: {label}")
        prepared.append((image, boxes))
    if not any(boxes for _, boxes in prepared):
        raise ValueError("val 정답 객체가 없습니다.")
    return prepared


def compare(args):
    for grid in (args.conf, args.nms):
        if not grid or any(not math.isfinite(v) or not 0 < v < 1 for v in grid):
            raise ValueError("신뢰도와 NMS IoU는 0과 1 사이여야 합니다.")
    confidences, nms_values = list(dict.fromkeys(args.conf)), list(dict.fromkeys(args.nms))
    if len(confidences) * len(nms_values) > 100 or args.imgsz < 32:
        raise ValueError("후보는 100개 이하, imgsz는 32 이상이어야 합니다.")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("기존 결과를 보존합니다. 새 출력 경로를 지정하세요.")
    images = validation_images(args.data)
    metadata = json.loads(args.weights.with_suffix(".json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if metadata.get("extension") != "camouflage" or metadata.get("classes") != [CLASS_NAME]:
        raise ValueError("보호색 학습 가중치와 JSON이 필요합니다.")
    if metadata.get("weights_sha256", digest) != digest:
        raise ValueError("가중치와 설정 JSON의 해시가 다릅니다.")
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    if model.names != {0: CLASS_NAME}:
        raise ValueError("가중치의 실제 클래스가 보호색 단일 클래스가 아닙니다.")
    candidates = []
    for nms in nms_values:
        totals = [dict(tp=0, fp=0, fn=0, duplicate_candidates=0) for _ in confidences]
        saturated = 0
        for index, (image, truth) in enumerate(images, 1):
            result = model.predict(str(image), conf=min(confidences), iou=nms, imgsz=args.imgsz,
                                   device=args.device, max_det=300, verbose=False)[0].boxes
            predictions = np.column_stack((result.xyxy.cpu().numpy(), result.conf.cpu().numpy()))
            saturated += int(len(predictions) >= 300)
            for conf, counts in zip(confidences, totals):
                for key, value in match_counts(truth, predictions, conf).items():
                    counts[key] += value
            if index % 50 == 0 or index == len(images):
                print(f"NMS {nms}: {index}/{len(images)} 이미지", flush=True)
        candidates.extend(dict(conf=conf, nms_iou=nms, capped_images=saturated, **metrics(counts))
                          for conf, counts in zip(confidences, totals))
    # 동률이면 사용자가 지정한 후보 순서를 보존합니다.
    best = max(candidates, key=lambda row: row["f1"])
    report = dict(split="val", images=len(images), weights_sha256=digest,
                  manifest_sha256=hashlib.sha256((args.data / "splits.json").read_bytes()).hexdigest(),
                  imgsz=args.imgsz, matching_iou=0.5, max_det=300, candidates=candidates, best=best,
                  tracking_evaluation="미실행", applied=False,
                  note="고정 임계값의 검출 P/R/F1입니다. mAP·추적 성능이 아니며 중복 후보는 FP의 부분집합입니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as folder:
        staged = Path(folder) / "report"
        staged.mkdir()
        (staged / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        with (staged / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)
        staged.rename(output)
    print(json.dumps(best, ensure_ascii=False, indent=2))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("datasets/camouflage"))
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--conf", type=float, nargs="+", default=[0.05, 0.2, 0.4, 0.5, 0.7])
    p.add_argument("--nms", type=float, nargs="+", default=[0.7, 0.5, 0.3])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    return p


if __name__ == "__main__":
    compare(parser().parse_args())
