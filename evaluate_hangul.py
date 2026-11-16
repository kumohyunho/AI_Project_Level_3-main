"""확장 기능(한글 자모) 평가: 추적 성능과 자모 분류 정확도.  설계: ``docs/HANGUL.md``

``hangul_dataset.py``가 만든 떠다니는 자모 영상(``tracking/<split>/<seq>``)에서 확장 기능(``extensions/hangul.py``)을 실행하고,
정답(ID, 자모, 회전 각도)과 비교합니다. 검출은 영상마다 한 번만 실행하고, 추적기 설정만 바꿔 가며 비교합니다.

1. **추적**: HOTA·IDF1·MOTA·IDSW·Frag (``metrics.py``, 보행자 평가와 같은 계산). 자모 종류와 무관하게 ID만 봅니다.
2. **자모 분류** (정답 박스와 IoU 0.5 이상으로 맞은 트랙 기준)
   * 프레임 단위: 그 프레임의 (높은 신뢰도) 검출 클래스가 정답과 같은 비율
   * 다수결(같은 프레임): 위와 같은 프레임에서 다수결 자모가 정답과 같은 비율 → **다수결의 효과 = 다수결 − 프레임 단위**
   * 화면 표시(전체): 높은 신뢰도 검출이 없던 프레임(가림 등)까지 포함해, 화면에 표시된 자모가 정답과 같은 비율
   * 트랙 단위: 영상이 끝났을 때 트랙의 최종 자모가 정답과 같은 비율
3. **회전 각도별**(0~15°, 15~30°, 30~45°) 검출률과 정확도, **자주 헷갈리는 자모 쌍**, **자모별** 검출률·정확도

예)
    python evaluate_hangul.py datasets/hangul/tracking/test --trackers baseline improved --name test
    python evaluate_hangul.py datasets/hangul/tracking/test --weights training_runs/hangul/weights/best.pt --device 0

결과: ``results/hangul/<name>/summary.md``·``summary.json``, 추적기별 ``confusion_<tracker>.csv``·``per_jamo_<tracker>.csv``,
MOT 형식 추적 결과 ``<tracker>/<seq>.txt``. 추적 지표는 ``results/comparison.json``에도 추가되어 GUI '성능 비교' 탭에 나타납니다.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from extensions.hangul import CHAR, KIND, HangulJamoExtension
from metrics import combine, evaluate_files
from mot_data import find_sequences
from pipeline import ROOT, needs_features
from results_io import write_mot
from tracker import extract_features
from tracker.matching import iou_matrix, linear_assignment

ANGLE_BINS = [(0, 15), (15, 30), (30, 45.01)]
MATCH_IOU = 0.5


def angle_bin(angle):
    for low, high in ANGLE_BINS:
        if low <= abs(angle) < high:
            return f"{low:g}~{min(high, 45):g}°"
    return f"{ANGLE_BINS[-1][1]:g}° 초과"


def load_truth(sequence):
    """``{frame: [(id, box(x1y1x2y2, 0부터), visibility)]}``, ``{id: 자모}``, ``{(frame, id): 각도}``."""
    folder = sequence.gt_path.parent
    objects = json.loads((folder / "objects.json").read_text(encoding="utf-8"))
    chars = {int(key): value["char"] for key, value in objects.items()}
    frames = defaultdict(list)
    for line in sequence.gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frame, track_id, left, top, width, height, _, _, visibility = line.split(",")
        x1, y1 = float(left) - 1, float(top) - 1  # MOT 좌표는 1부터
        frames[int(frame)].append((int(track_id), [x1, y1, x1 + float(width), y1 + float(height)], float(visibility)))
    angles = {}
    path = folder / "angles.csv"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            frame, track_id, angle = line.split(",")
            angles[(int(frame), int(track_id))] = float(angle)
    return frames, chars, angles


def detect_sequence(extension, sequence):
    """영상 전체를 한 번 검출합니다. ``{frame: (boxes, scores, class_ids, features)}``와 프레임당 검출 시간(ms)."""
    detections, elapsed = {}, 0.0
    for number in sequence.frames:
        frame = sequence.read(number)
        started = perf_counter()
        boxes, scores, class_ids = extension.detect(frame)
        elapsed += perf_counter() - started
        features = extract_features(frame, boxes) if len(boxes) else []
        detections[number] = (boxes, scores, class_ids, features)
        if number % 50 == 0 or number == sequence.length:
            print(f"{sequence.name}: 검출 {number}/{sequence.length}", flush=True)
    return detections, elapsed * 1000 / max(1, sequence.length)


def run_tracker(extension, sequence, detections, tracker):
    """같은 검출 위에서 추적·투표만 실행합니다.

    반환: MOT 결과 행, ``{(frame, id): (박스, 표시 자모, 그 프레임 검출 자모)}``, ``{id: 최종 자모}``, 프레임당 추적 시간(ms)
    """
    extension.fps = sequence.fps
    extension.reset(tracker)
    use_features = needs_features(extension.tracker_config)
    rows, shown, elapsed = [], {}, 0.0
    for number in sequence.frames:
        boxes, scores, class_ids, features = detections[number]
        started = perf_counter()
        tracked = extension.track(boxes, scores, class_ids, features if use_features and len(boxes) else None,
                                  (sequence.width, sequence.height))
        elapsed += perf_counter() - started
        for track, voted, observed in tracked:
            box = [float(v) for v in track.box]
            rows.append((number, track.track_id, *box, track.score))
            shown[(number, track.track_id)] = (box, CHAR.get(voted), CHAR.get(observed))
    final = {track_id: CHAR.get(extension.votes.best(track_id)) for track_id in extension.votes.scores}
    return rows, shown, final, elapsed * 1000 / max(1, sequence.length)


def classify(sequence, truth, shown, final):
    """정답 박스와 트랙을 프레임마다 1:1로 맞추고(IoU ≥ 0.5) 자모 분류 기록을 만듭니다."""
    frames, chars, angles = truth
    by_frame = defaultdict(list)
    for (frame, track_id), value in shown.items():
        by_frame[frame].append((track_id, *value))
    records = []
    for frame in sequence.frames:
        truths, tracked = frames.get(frame, []), by_frame.get(frame, [])
        matched = {}
        if truths and tracked:
            overlap = iou_matrix([t[1] for t in truths], [t[1] for t in tracked])
            pairs, _, _ = linear_assignment(1.0 - overlap, 1.0 - MATCH_IOU)
            matched = dict(pairs)
        for row, (gt_id, _, visibility) in enumerate(truths):
            angle = angles.get((frame, gt_id), 0.0)
            record = dict(sequence=sequence.name, frame=frame, gt_id=gt_id, char=chars[gt_id], angle=angle,
                          visibility=visibility, matched=row in matched, track_id=None, shown=None,
                          observed=None, final=None)
            if row in matched:
                track_id, _, voted, observed = tracked[matched[row]]
                record.update(track_id=track_id, shown=voted, observed=observed, final=final.get(track_id))
            records.append(record)
    return records


def ratio(hit, total):
    return 100.0 * hit / total if total else float("nan")


def summarize(records):
    """분류 기록에서 정확도·각도별·자모별·혼동 쌍 통계를 냅니다."""
    matched = [r for r in records if r["matched"]]
    with_shown = [r for r in matched if r["shown"]]
    with_observed = [r for r in matched if r["observed"]]
    summary = {
        "gt_boxes": len(records), "matched": len(matched),
        "detection_rate": ratio(len(matched), len(records)),
        "frame_accuracy": ratio(sum(r["observed"] == r["char"] for r in with_observed), len(with_observed)),
        "shown_accuracy": ratio(sum(r["shown"] == r["char"] for r in with_shown), len(with_shown)),
        # 다수결의 효과는 같은 프레임(투표한 검출이 있는 프레임)끼리 비교해야 공정합니다.
        "shown_accuracy_voting_frames": ratio(sum(r["shown"] == r["char"] for r in with_observed), len(with_observed)),
        "voting_frames": len(with_observed),
        "final_accuracy": ratio(sum(r["final"] == r["char"] for r in with_shown), len(with_shown)),
    }
    # 트랙 단위: 트랙이 가장 오래 따라간 정답 객체의 자모와 최종 자모가 같은가
    per_track = defaultdict(Counter)
    for r in with_shown:
        per_track[(r["sequence"], r["track_id"])][(r["char"], r["final"])] += 1
    tracks_ok = sum(max(c, key=c.get)[0] == max(c, key=c.get)[1] for c in per_track.values())
    summary["tracks"] = len(per_track)
    summary["track_accuracy"] = ratio(tracks_ok, len(per_track))

    summary["by_angle"] = {}
    for low, high in ANGLE_BINS:
        name = f"{low:g}~{min(high, 45):g}°"
        group = [r for r in records if low <= abs(r["angle"]) < high]
        hit = [r for r in group if r["matched"]]
        seen = [r for r in hit if r["observed"]]
        shown = [r for r in hit if r["shown"]]
        summary["by_angle"][name] = dict(
            gt_boxes=len(group), detection_rate=ratio(len(hit), len(group)),
            frame_accuracy=ratio(sum(r["observed"] == r["char"] for r in seen), len(seen)),
            shown_accuracy=ratio(sum(r["shown"] == r["char"] for r in shown), len(shown)))
    summary["by_kind"] = {}
    for kind in ("자음", "모음"):
        group = [r for r in records if KIND_BY_CHAR[r["char"]] == kind]
        hit = [r for r in group if r["matched"] and r["shown"]]
        summary["by_kind"][kind] = dict(
            gt_boxes=len(group), detection_rate=ratio(sum(r["matched"] for r in group), len(group)),
            shown_accuracy=ratio(sum(r["shown"] == r["char"] for r in hit), len(hit)))

    per_jamo = []
    for char in sorted({r["char"] for r in records}, key=list(CHAR.values()).index):
        group = [r for r in records if r["char"] == char]
        hit = [r for r in group if r["matched"] and r["shown"]]
        seen = [r for r in group if r["matched"] and r["observed"]]
        per_jamo.append(dict(
            char=char, kind=KIND_BY_CHAR[char], gt_boxes=len(group),
            detection_rate=ratio(sum(r["matched"] for r in group), len(group)),
            frame_accuracy=ratio(sum(r["observed"] == char for r in seen), len(seen)),
            shown_accuracy=ratio(sum(r["shown"] == char for r in hit), len(hit))))
    confusion_frame = Counter((r["char"], r["observed"]) for r in with_observed if r["observed"] != r["char"])
    confusion_shown = Counter((r["char"], r["shown"]) for r in with_shown if r["shown"] != r["char"])
    return summary, per_jamo, confusion_frame, confusion_shown


KIND_BY_CHAR = {CHAR[name]: kind for name, kind in KIND.items()}


def fmt(value, digits=1):
    return "—" if value != value else f"{value:.{digits}f}"   # NaN → —


def write_report(folder, name, weights, sequences, results, det_ms):
    lines = [f"# 한글 자모 평가: {name}", "",
             f"* 가중치: `{weights}`", f"* 영상: {', '.join(s.name for s in sequences)} "
             f"(총 {sum(s.length for s in sequences)}프레임)", f"* 프레임당 검출 시간: {det_ms:.1f} ms", "",
             "## 추적 성능", "",
             "| 추적기 | HOTA | IDF1 | MOTA | IDSW | Frag | 정답 ID 수 | 만든 ID 수 | 추적 ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tracker, r in results.items():
        m = r["tracking"]
        lines.append(f"| {tracker} | {m['HOTA']:.2f} | {m['IDF1']:.2f} | {m['MOTA']:.2f} | {m['IDSW']} | {m['Frag']} | "
                     f"{m['GT_IDs']} | {m['IDs']} | {r['track_ms']:.2f} |")
    lines += ["", "## 자모 분류 정확도 (%)", "",
              "정답 박스와 IoU 0.5 이상으로 맞은 트랙 기준. 프레임 단위 = 그 프레임의 높은 신뢰도 검출 클래스, "
              "다수결 = 같은 프레임에서 그때까지의 다수결 자모, 다수결 효과 = 다수결 − 프레임 단위, "
              "화면 표시 = 높은 신뢰도 검출이 없던 프레임까지 포함한 전체, 트랙 단위 = 트랙의 최종 자모.", "",
              "| 추적기 | 검출률 | 프레임 단위 | 다수결(같은 프레임) | 다수결 효과 | 화면 표시(전체) | 트랙 단위 | 트랙 수 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tracker, r in results.items():
        c = r["classification"]
        gain = c["shown_accuracy_voting_frames"] - c["frame_accuracy"]
        lines.append(f"| {tracker} | {fmt(c['detection_rate'])} | {fmt(c['frame_accuracy'])} | "
                     f"{fmt(c['shown_accuracy_voting_frames'])} | {'—' if gain != gain else f'{gain:+.1f}'} | "
                     f"{fmt(c['shown_accuracy'])} | {fmt(c['track_accuracy'])} | {c['tracks']} |")
    for tracker, r in results.items():
        c = r["classification"]
        lines += ["", f"## {tracker}: 회전 각도별", "",
                  "| 회전 각도(절댓값) | 정답 박스 | 검출률 | 프레임 단위 | 화면 표시 |", "|---|---:|---:|---:|---:|"]
        for angle, v in c["by_angle"].items():
            lines.append(f"| {angle} | {v['gt_boxes']} | {fmt(v['detection_rate'])} | {fmt(v['frame_accuracy'])} | "
                         f"{fmt(v['shown_accuracy'])} |")
        lines += ["", "| 종류 | 정답 박스 | 검출률 | 화면 표시 |", "|---|---:|---:|---:|"]
        for kind, v in c["by_kind"].items():
            lines.append(f"| {kind} | {v['gt_boxes']} | {fmt(v['detection_rate'])} | {fmt(v['shown_accuracy'])} |")
        lines += ["", f"### {tracker}: 자주 헷갈리는 자모 (정답 → 판정, 프레임 수)", ""]
        frame_pairs = ", ".join(f"{a}→{b} {n}" for (a, b), n in r["confusion_frame"].most_common(10)) or "없음"
        shown_pairs = ", ".join(f"{a}→{b} {n}" for (a, b), n in r["confusion_shown"].most_common(10)) or "없음"
        lines += [f"* 프레임 단위: {frame_pairs}", f"* 화면 표시(다수결): {shown_pairs}"]
        worst = sorted((j for j in r["per_jamo"] if j["gt_boxes"]),
                       key=lambda j: (j["detection_rate"] if j["detection_rate"] == j["detection_rate"] else 0))[:5]
        lines += ["", f"### {tracker}: 검출률이 낮은 자모", "",
                  ", ".join(f"{j['char']} {fmt(j['detection_rate'])}%" for j in worst)]
    lines += ["", "검출률 = 정답 박스 중 트랙과 맞은 비율. 이 영상에 나오지 않은 자모는 표에 없습니다.", ""]
    (folder / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def main(args):
    sequences = find_sequences(args.sequences)
    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"가중치가 없습니다: {weights} (colab/train_hangul.ipynb로 학습하거나 --weights로 지정)")

    class Evaluated(HangulJamoExtension):
        imgsz = args.imgsz

        @classmethod
        def weights_path(cls):
            return weights

    extension = Evaluated(device=args.device, fps=sequences[0].fps)
    folder = ROOT / "results" / "hangul" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    truths = {s.name: load_truth(s) for s in sequences}
    cached, det_ms = {}, []
    for sequence in sequences:
        cached[sequence.name], ms = detect_sequence(extension, sequence)
        det_ms.append(ms)
    det_ms = float(np.mean(det_ms))

    results, comparison = {}, []
    for tracker in args.trackers:
        raw, records, track_ms = [], [], []
        for sequence in sequences:
            rows, shown, final, ms = run_tracker(extension, sequence, cached[sequence.name], tracker)
            track_ms.append(ms)
            target = folder / tracker / f"{sequence.name}.txt"
            write_mot(target, rows)
            raw.append(evaluate_files(sequence.gt_path, target, sequence.frames))
            records += classify(sequence, truths[sequence.name], shown, final)
        tracking = combine(raw)
        classification, per_jamo, confusion_frame, confusion_shown = summarize(records)
        results[tracker] = dict(tracking=tracking, classification=classification, per_jamo=per_jamo,
                                confusion_frame=confusion_frame, confusion_shown=confusion_shown,
                                track_ms=float(np.mean(track_ms)), config=extension.tracker_config.to_dict())
        with (folder / f"per_jamo_{tracker}.csv").open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(per_jamo[0]) if per_jamo else ["char"])
            writer.writeheader()
            writer.writerows(per_jamo)
        with (folder / f"confusion_{tracker}.csv").open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["기준", "정답", "판정", "프레임 수"])
            for label, counter in (("프레임 단위", confusion_frame), ("화면 표시", confusion_shown)):
                writer.writerows([label, a, b, n] for (a, b), n in counter.most_common())
        total = det_ms + results[tracker]["track_ms"]
        comparison.append(dict(name=f"한글 자모 {args.name}", weights=weights.name, tracker=tracker,
                               sequences=[s.name for s in sequences], det_ms=det_ms,
                               track_ms=results[tracker]["track_ms"], total_ms=total, fps=1000 / max(total, 1e-6),
                               created=datetime.now().isoformat(timespec="seconds"),
                               config=results[tracker]["config"], **tracking))

    print(write_report(folder, args.name, weights, sequences, results, det_ms))
    (folder / "summary.json").write_text(json.dumps(dict(
        name=args.name, weights=str(weights), sequences=[s.name for s in sequences], det_ms=det_ms,
        created=datetime.now().isoformat(timespec="seconds"),
        results={t: dict(tracking=r["tracking"], classification=r["classification"], track_ms=r["track_ms"],
                         config=r["config"], per_jamo=r["per_jamo"],
                         confusion_frame=[[a, b, n] for (a, b), n in r["confusion_frame"].most_common()],
                         confusion_shown=[[a, b, n] for (a, b), n in r["confusion_shown"].most_common()])
                 for t, r in results.items()}), ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_comparison:
        from run_mot import save_comparison

        save_comparison(comparison)
    print(f"\n결과 폴더: {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("sequences", nargs="+", help="예: datasets/hangul/tracking/test")
    parser.add_argument("--trackers", nargs="+", default=["baseline", "improved"],
                        help="baseline, improved, baseline+appearance 등, configs/<이름>.json의 이름")
    parser.add_argument("--weights", default=str(ROOT / "models" / "hangul_jamo.pt"))
    parser.add_argument("--name", default="test", help="결과 폴더 이름 (results/hangul/<name>)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-comparison", action="store_true", help="results/comparison.json에 추가하지 않음")
    main(parser.parse_args())
