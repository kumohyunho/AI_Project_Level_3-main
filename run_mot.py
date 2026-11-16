"""MOT17/MOT20 시퀀스에서 추적기 설정별 성능을 측정하고 비교표를 만듭니다 (요구 사항 3·7·9).

예)
    python run_mot.py datasets/mot/tracking/test --trackers baseline improved tuned --name test
    python run_mot.py datasets/mot/tracking/test --weights models/person.pt --trackers improved --name crowd

결과:
    results/mot/<name>/<tracker>/<seq>.txt   추적 결과 (MOT 형식, TrackEval에도 사용 가능)
    results/comparison.json                  GUI '성능 비교' 탭이 읽는 비교표
    results/comparison.md                    보고서용 표
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from metrics import combine, evaluate_files
from mot_data import find_sequences, load_detections
from pipeline import ROOT, default_weights, tracker_config
from results_io import write_mot
from tracker import ByteTracker

COMPARISON = ROOT / "results" / "comparison.json"


def run_tracker(config, sequence, detections):
    """캐시된 검출 위에서 추적기만 실행합니다. (결과 행 목록, 프레임당 추적 시간 ms)"""
    tracker = ByteTracker(config, sequence.fps)
    rows, elapsed = [], 0.0
    empty = (np.zeros((0, 4)), np.zeros(0), [])
    use_features = config.appearance or config.recover_lost or config.reid_buffer > 0
    for number in sequence.frames:
        boxes, scores, features = detections.get(number, empty)
        started = perf_counter()
        tracks = tracker.update(boxes, scores, np.zeros(len(boxes), dtype=int),
                                features if use_features else None, (sequence.width, sequence.height))
        elapsed += perf_counter() - started
        rows += [(number, t.track_id, *t.box, t.score) for t in tracks]
    return rows, elapsed * 1000 / max(1, sequence.length)


def evaluate_config(config, sequences, cached, output_dir=None, events=None):
    raw, track_ms = [], []
    for sequence in sequences:
        detections, _ = cached[sequence.name]
        rows, ms = run_tracker(config, sequence, detections)
        track_ms.append(ms)
        target = (output_dir or ROOT / "results" / "tmp") / f"{sequence.name}.txt"
        write_mot(target, rows)
        sequence_events = [] if events is not None else None
        raw.append(evaluate_files(sequence.gt_path, target, sequence.frames, sequence_events))
        if events is not None:
            events.extend(dict(e, sequence=sequence.name) for e in sequence_events)
    return combine(raw), float(np.mean(track_ms))


def save_comparison(entries):
    COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(COMPARISON.read_text(encoding="utf-8")) if COMPARISON.is_file() else []
    keys = {(e["name"], e["tracker"], e["weights"]) for e in entries}
    merged = [e for e in existing if (e["name"], e["tracker"], e["weights"]) not in keys] + entries
    COMPARISON.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| 실험 | 가중치 | 추적기 | HOTA | IDF1 | MOTA | IDSW | Frag | FPS | 평균 ms |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e in merged:
        lines.append(f"| {e['name']} | {e['weights']} | {e['tracker']} | {e['HOTA']:.2f} | "
                     f"{e['IDF1']:.2f} | {e['MOTA']:.2f} | {e['IDSW']} | {e['Frag']} | "
                     f"{e['fps']:.1f} | {e['total_ms']:.1f} |")
    COMPARISON.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(args):
    sequences = find_sequences(args.sequences)
    weights = args.weights or default_weights()
    cached = {s.name: load_detections(s, weights, args.imgsz, args.device) for s in sequences}
    det_ms = float(np.mean([cached[s.name][1] for s in sequences]))
    entries = []
    for name in args.trackers:
        config = tracker_config(name)
        output = ROOT / "results" / "mot" / args.name / name
        metrics, track_ms = evaluate_config(config, sequences, cached, output)
        total = det_ms + track_ms
        entries.append(dict(name=args.name, weights=Path(weights).name, tracker=name,
                            sequences=[s.name for s in sequences], det_ms=det_ms,
                            track_ms=track_ms, total_ms=total, fps=1000 / max(total, 1e-6),
                            created=datetime.now().isoformat(timespec="seconds"),
                            config=config.to_dict(), **metrics))
        print(f"{name}: HOTA {metrics['HOTA']:.2f} IDF1 {metrics['IDF1']:.2f} "
              f"MOTA {metrics['MOTA']:.2f} IDSW {metrics['IDSW']}")
    save_comparison(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("sequences", nargs="+", help="시퀀스 폴더 또는 시퀀스들의 상위 폴더")
    parser.add_argument("--trackers", nargs="+", default=["baseline", "improved"],
                        help="baseline, improved, configs/<이름>.json의 이름")
    parser.add_argument("--weights", help="사람 검출 가중치 (기본: models/person.pt → yolo11n.pt)")
    parser.add_argument("--name", default="experiment", help="비교표에 표시할 실험 이름")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args())

