"""보호색 추적 파라미터 검증(val) 탐색. 같은 검출 결과로 후보를 비교합니다.

학습된 전용 YOLO와 ID가 확인된 MoCA val 시퀀스가 필요합니다.
test 분할은 튜닝에 사용하지 않습니다. 자세한 방법: docs/CAMOUFLAGE_TUNING.md.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from evaluate_camouflage import load_evaluation_annotations
from extensions.camouflage import default_tracker, read_tracker_config, validate_tracker
from metrics import Frame, combine, evaluate_sequence, load_ground_truth
from mot_data import Sequence, find_sequences
from pipeline import needs_features
from run_camouflage import ROOT, make_extension
from tracker import ByteTracker, extract_features

METRICS = ("HOTA", "IDF1", "MOTA", "IDSW", "Frag", "FP", "FN")


@dataclass
class CachedSequence:
    sequence: Sequence
    evaluated_frames: list
    ground_truth: dict
    detections: list
    image_digest: str


def candidates(base, high_values, match_values, buffer_values):
    """기존 설정을 반드시 포함하며 중복 후보는 제거합니다. 동점이면 기존 설정을 우선합니다."""
    if not all((high_values, match_values, buffer_values)):
        raise ValueError("탐색 값은 비어 있으면 안 됩니다.")
    if math.prod(map(len, (high_values, match_values, buffer_values))) > 255:
        raise ValueError("격자 후보는 최대 255개까지 지정하세요 (기존 설정 포함 최대 256개).")
    result = [validate_tracker(base.to_dict())]
    seen = {json.dumps(base.to_dict(), sort_keys=True)}
    for high, match, buffer in itertools.product(high_values, match_values, buffer_values):
        config = replace(base, high_thresh=high, new_track_thresh=min(1.0, high+0.1),
                         match_thresh=match, track_buffer=buffer)
        validate_tracker(config.to_dict())
        key = json.dumps(config.to_dict(), sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(config)
    return result


def ground_truth_for_tuning(sequence, frames):
    for line in sequence.gt_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split(",")))
        if (len(values) != 9 or not all(math.isfinite(v) for v in values)
                or any(v != int(v) for v in values[:2]) or values[0] not in frames
                or values[6:8] != [1, 1] or not 0 <= values[8] <= 1):
            raise ValueError("MOT 프레임·ID·플래그가 잘못되었습니다. #8 검사기를 먼저 실행하세요.")
    truth, _ = load_ground_truth(sequence.gt_path, set(frames))
    for frame in truth.values():
        if len(set(frame.ids)) != len(frame.ids) or np.any(frame.ids < 1):
            raise ValueError("정답 ID가 중복되거나 양수가 아닙니다. #8 검사기를 먼저 실행하세요.")
        boxes = frame.boxes
        if (not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2])
                or np.any(boxes[:, :2] < 1) or np.any(boxes[:, 2] > sequence.width+1.001)
                or np.any(boxes[:, 3] > sequence.height+1.001)):
            raise ValueError("MOT 정답 박스가 올바르지 않습니다. #8 검사기를 먼저 실행하세요.")
    return truth


def cache_detections(sequences, annotations, truths, detector, low_thresh, with_features):
    """한 실행 안에서만 재사용합니다. 디스크 캐시나 프레임 이미지를 메모리에 보관하지 않습니다."""
    cached = []
    for sequence in sequences:
        frames, digest = [], hashlib.sha256()
        for number in sequence.frames:
            image = sequence.read(number)
            if image.shape[:2] != (sequence.height, sequence.width):
                raise ValueError(f"프레임 크기가 명세와 다릅니다: {sequence.name}/{number}")
            digest.update(image.tobytes())
            boxes, scores, classes = detector(image, low_thresh)
            boxes = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
            scores = np.asarray(scores, dtype=float).reshape(-1).copy()
            classes = np.asarray(classes).reshape(-1).copy()
            if (len(boxes) != len(scores) or len(scores) != len(classes)
                    or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
                    or not np.isfinite(classes).all() or np.any(scores < 0) or np.any(scores > 1)
                    or np.any(classes < 0) or np.any(classes != classes.astype(int))
                    or np.any(boxes[:, 2:] <= boxes[:, :2])):
                raise ValueError("검출 결과의 좌표·신뢰도·클래스 형식이 잘못되었습니다.")
            features = extract_features(image, boxes) if with_features else None
            frames.append((boxes, scores, classes.astype(int), features))
        cached.append(CachedSequence(sequence, annotations[sequence.name]["evaluated_frames"],
                                     truths[sequence.name], frames, digest.hexdigest()))
    return cached


def score_config(config, cached):
    """각 후보·영상마다 새 ByteTrack. 전 프레임 추적 후 주석 프레임에서만 평가합니다."""
    raw = []
    for item in cached:
        seq = item.sequence
        tracker = ByteTracker(config, seq.fps)
        predictions = {}
        evaluated = set(item.evaluated_frames)
        for number, (boxes, scores, classes, features) in enumerate(item.detections, 1):
            tracks = tracker.update(boxes, scores, classes, features, (seq.width, seq.height))
            if number in evaluated:
                # 기존 metrics.load_ground_truth는 MOT의 1 기반 좌표를 유지합니다.
                # 추적기의 0 기반 박스에 +1하여 같은 좌표계에서 비교합니다.
                predictions[number] = Frame(np.array([t.track_id for t in tracks], dtype=int),
                                            np.array([t.box+1 for t in tracks]).reshape(-1, 4))
        raw.append(evaluate_sequence(item.ground_truth, predictions, item.evaluated_frames))
    return combine(raw)


def ranking(row):
    metrics = row["metrics"]
    return (-metrics["HOTA"], -metrics["IDF1"], -metrics["MOTA"], metrics["IDSW"], row["candidate"])


def tune(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"다른 출력 경로를 지정하세요: {output}")
    sequences = find_sequences([args.sequences])
    annotations = load_evaluation_annotations(sequences, expected_split="val")
    if output.is_relative_to(Path(args.sequences).resolve()):
        raise ValueError("입력 시퀀스 내부에는 결과를 쓰지 않습니다.")
    truths = {seq.name: ground_truth_for_tuning(seq, annotations[seq.name]["evaluated_frames"]) for seq in sequences}
    if not any(len(frame.ids) for truth in truths.values() for frame in truth.values()):
        raise ValueError("검증 정답에 객체가 없어 파라미터를 선택할 수 없습니다.")
    # 잘못된 모델 경로/탐색 설정은 추론 전에 중단합니다.
    weights = Path(args.weights).resolve()
    weights_digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    metadata_path = weights.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "weights_sha256" in metadata and metadata["weights_sha256"] != weights_digest:
        raise ValueError("학습 설정 JSON의 가중치 해시가 현재 파일과 다릅니다.")
    base = validate_tracker(read_tracker_config(args.base_config, weights) if args.base_config
                            else metadata.get("tracker", default_tracker().to_dict()))
    configs = candidates(base, args.high, args.match, args.buffer)
    extension = make_extension(weights, args.device, sequences[0].fps)
    try:
        cached = cache_detections(sequences, annotations, truths, extension.detector,
                                  base.low_thresh, needs_features(base))
    finally:
        extension.close()
    rows = []
    for index, config in enumerate(configs):
        scores = score_config(config, cached)
        rows.append(dict(candidate=index, tracker=config.to_dict(), metrics=scores))
    ordered = sorted(rows, key=ranking)
    best, original = ordered[0], rows[0]
    provenance = [dict(sequence=item.sequence.name, folder=str(item.sequence.folder),
                       fps=item.sequence.fps, frames=item.sequence.length,
                       evaluated_frames=item.evaluated_frames, decoded_images_sha256=item.image_digest,
                       gt_sha256=hashlib.sha256(item.sequence.gt_path.read_bytes()).hexdigest(),
                       annotations_sha256=hashlib.sha256((item.sequence.folder/"annotations.json").read_bytes()).hexdigest())
                  for item in cached]
    summary = dict(extension="camouflage", split="val", weights=str(weights), weights_sha256=weights_digest,
                   model_metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest())
    summary.update(objective="HOTA 최대 → IDF1 최대 → MOTA 최대 → IDSW 최소 → 기존 설정/후보 순서",
                   baseline=original, best=best, candidates=ordered, sequences=provenance,
                   changed=best["tracker"] != original["tracker"],
                   delta={key: best["metrics"][key]-original["metrics"][key] for key in METRICS},
                   protocol="동일 검출·동일 프레임. 모든 프레임 추적, 주석 프레임만 평가. val에서만 선택.",
                   limitations=["현재 검증 데이터와 후보 격자 안의 최적값이며 다른 영상의 개선을 보장하지 않음",
                                "검출 결과를 재사용한 탐색이므로 실시간 FPS를 측정하지 않음",
                                "별도의 test 평가·종/보호색 여부·ID 의미 검증은 수행하지 않음"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as folder:
        staging = Path(folder)/"tuning"
        staging.mkdir()
        (staging/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        selection = dict(extension="camouflage", weights_sha256=weights_digest, split="val", tracker=best["tracker"],
                         val_metrics=best["metrics"], candidate=best["candidate"],
                         status="검증 데이터 후보 선택 완료·별도 test 평가 필요")
        (staging/"best_config.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
        columns = ["candidate", "high_thresh", "new_track_thresh", "match_thresh", "track_buffer", *METRICS]
        with (staging/"comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(dict(candidate=row["candidate"], **{k:row["tracker"][k] for k in columns[1:5]},
                                 **{k:row["metrics"][k] for k in METRICS}) for row in ordered)
        lines = ["# 보호색 추적 검증 설정 비교", "", summary["protocol"], "",
                 "| 후보 | high | new | match | buffer | HOTA | IDF1 | MOTA | IDSW |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for row in ordered:
            c, m = row["tracker"], row["metrics"]
            lines.append(f"| {row['candidate']} | {c['high_thresh']} | {c['new_track_thresh']} | {c['match_thresh']} | "
                         f"{c['track_buffer']} | {m['HOTA']:.3f} | {m['IDF1']:.3f} | {m['MOTA']:.3f} | {m['IDSW']} |")
        lines.extend(["", "후보 0은 기존 설정. 표의 값은 이 실행의 입력 데이터에 대한 값입니다.", *summary["limitations"]])
        (staging/"comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        staging.rename(output)
    print(f"기존 후보 0 → 선택 후보 {best['candidate']}, 변경 여부 {summary['changed']}. 결과: {output}")
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sequences", type=Path, help="datasets/camouflage/tracking/val")
    p.add_argument("--weights", type=Path, default=ROOT/"models"/"camouflage.pt")
    p.add_argument("--base-config", type=Path, help="생략 시 가중치와 함께 저장된 추적 설정")
    p.add_argument("--high", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    p.add_argument("--match", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    p.add_argument("--buffer", type=int, nargs="+", default=[30, 60, 90])
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=ROOT/"results"/"camouflage"/"tuning")
    return p


if __name__ == "__main__":
    tune(parser().parse_args())
