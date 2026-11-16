"""MoCA의 주석 프레임에서만 HOTA·IDF1·MOTA·IDSW 평가. 전체 프레임은 순서대로 추적합니다."""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

from extensions.camouflage import read_tracker_config
from metrics import combine, evaluate_files
from mot_data import find_sequences
from results_io import write_mot
from run_camouflage import ROOT, make_extension


def load_evaluation_annotations(sequences, expected_split=None):
    """평가·튜닝이 같은 주석 프레임 검사를 사용합니다. 튜닝은 val만 허용합니다."""
    annotations = {}
    # 정답 ID가 없는 영상을 1번 개체로 간주해 성능을 부풀리지 않습니다.
    for seq in sequences:
        if min(seq.length, seq.width, seq.height) < 1 or not math.isfinite(seq.fps) or seq.fps <= 0:
            raise ValueError(f"{seq.name}: 시퀀스 길이·크기·FPS가 올바르지 않습니다.")
        info = json.loads((seq.folder / "annotations.json").read_text(encoding="utf-8"))
        if info.get("ids_verified") is not True or not seq.gt_path.is_file():
            raise ValueError(f"{seq.name}: 추적 ID 정답이 없습니다. ID 주석 또는 단일 개체 확인이 필요합니다.")
        frames = info["evaluated_frames"]
        if (not isinstance(frames, list) or not frames
                or any(type(frame) is not int for frame in frames)
                or frames != sorted(set(frames)) or not set(frames) <= set(seq.frames)):
            raise ValueError(f"{seq.name}: 평가 프레임 목록이 올바르지 않습니다.")
        if seq.name in annotations:
            raise ValueError(f"중복 시퀀스 이름: {seq.name}")
        if expected_split is not None and (info.get("split") != expected_split or seq.folder.parent.name != expected_split):
            raise ValueError(f"{seq.name}: 튜닝에는 {expected_split} 분할만 사용하세요. test/train 사용은 금지합니다.")
        annotations[seq.name] = info
    return annotations


def evaluate(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"다른 출력 폴더를 지정하세요: {output}")
    sequences = find_sequences([args.sequences])
    annotations = load_evaluation_annotations(sequences)
    config = read_tracker_config(args.tracker_config, args.weights) if args.tracker_config else None
    output.mkdir(parents=True)
    raw, per_sequence, all_events = [], [], []
    for seq in sequences:
        extension = make_extension(args.weights, args.device, seq.fps, config)
        rows, elapsed = [], 0.0
        try:
            for number in seq.frames:
                image = seq.read(number)
                started = perf_counter()
                tracks = extension.process(image, number)
                elapsed += perf_counter() - started
                rows.extend((number, t.track_id, *t.box, t.score) for t in tracks)
        finally:
            extension.close()
        result = output / f"{seq.name}.txt"
        write_mot(result, rows)
        events = []
        value = evaluate_files(seq.gt_path, result, annotations[seq.name]["evaluated_frames"], events)
        raw.append(value)
        per_sequence.append(dict(sequence=seq.name, metrics=combine([value]),
                                 annotated_frames=len(annotations[seq.name]["evaluated_frames"]),
                                 processed_frames=seq.length, processing_ms=elapsed * 1000 / seq.length))
        all_events.extend(dict(sequence=seq.name, **event) for event in events)
    summary = dict(weights=str(Path(args.weights).resolve()), tracker=extension.tracker_config.to_dict(),
                   metrics=combine(raw), sequences=per_sequence, events=all_events,
                   protocol="전 프레임 추적, annotations.json의 주석 프레임만 평가. 보간 정답 생성 안 함.")
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sequences", type=Path, help="datasets/camouflage/tracking/test 등")
    p.add_argument("--weights", type=Path, default=ROOT / "models" / "camouflage.pt")
    p.add_argument("--tracker-config", type=Path, help="검증 영상으로 조정한 TrackerConfig JSON")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "camouflage" / "evaluation")
    return p


if __name__ == "__main__":
    evaluate(parser().parse_args())
