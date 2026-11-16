"""가림·교차 상황의 추적 실패 사례 분석 (요구 사항 4·7).

ID 변경(IDSW)과 추적 끊김(Frag)이 일어난 프레임을 찾아, 그 순간 정답 박스의 상태로 원인을 분류합니다.

* 교차: 다른 정답 박스와 IoU 0.3 이상으로 겹침 (사람끼리 스쳐 지나감)
* 가림: 정답의 visibility < 0.5 (MOT17/20 정답에 포함된 가시 비율)
* 화면 경계: 박스가 화면 가장자리 3% 이내
* 기타: 위 조건이 없음 (검출 누락, 급격한 움직임 등)

결과: ``results/failures/<name>/events.csv``, ``summary.md``, 사례 이미지(``--images``).

예)
    python analyze_failures.py datasets/mot/tracking/test --trackers baseline improved --images 12
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from metrics import load_ground_truth
from mot_data import find_sequences, load_detections
from pipeline import ROOT, default_weights, tracker_config
from run_mot import evaluate_config
from tracker.matching import iou_matrix


def classify(event, gt, sequence):
    frame = gt.get(event["frame"])
    if frame is None or event["gt_id"] not in frame.ids:
        return "기타", 0.0, 1.0
    index = int(np.flatnonzero(frame.ids == event["gt_id"])[0])
    box = frame.boxes[index]
    overlap = iou_matrix(box[None], np.delete(frame.boxes, index, axis=0))
    max_overlap = float(overlap.max()) if overlap.size else 0.0
    visibility = float(frame.visibility[index])
    x1, y1, x2, y2 = box
    margin_x, margin_y = 0.03 * sequence.width, 0.03 * sequence.height
    border = sequence.width and (x1 <= margin_x or y1 <= margin_y or
                                 x2 >= sequence.width - margin_x or y2 >= sequence.height - margin_y)
    if max_overlap >= 0.3:
        return "교차", max_overlap, visibility
    if visibility < 0.5:
        return "가림", max_overlap, visibility
    if border:
        return "화면 경계", max_overlap, visibility
    return "기타", max_overlap, visibility


def draw_case(sequence, event, gt, result_path, target):
    """사례 프레임에 정답(흰색)과 추적 결과(이전 ID 빨강, 새 ID 초록)를 그려 저장합니다."""
    import cv2

    image = sequence.read(event["frame"])
    frame = gt.get(event["frame"])
    if frame is not None:
        for gid, box in zip(frame.ids, frame.boxes):
            thickness = 3 if gid == event["gt_id"] else 1
            cv2.rectangle(image, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (255, 255, 255), thickness)
    with open(result_path, encoding="utf-8") as source:
        for line in source:
            values = line.split(",")
            if int(values[0]) != event["frame"]:
                continue
            tid = int(values[1])
            if tid not in (event["old_id"], event["new_id"]):
                continue
            x, y, w, h = map(float, values[2:6])
            color = (0, 0, 255) if tid == event["old_id"] else (0, 200, 0)
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            cv2.putText(image, f"ID {tid}", (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    label = f"{event['type']} {event['category']} frame {event['frame']} GT {event['gt_id']}: {event['old_id']}->{event['new_id']}"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", image)[1].tofile(str(target))


def main(args):
    sequences = find_sequences(args.sequences)
    weights = args.weights or default_weights()
    cached = {s.name: load_detections(s, weights, args.imgsz, args.device) for s in sequences}
    output = ROOT / "results" / "failures" / args.name
    summary = ["# 추적 실패 사례 분석", "",
               f"- 시퀀스: {', '.join(s.name for s in sequences)}", f"- 검출 가중치: {Path(weights).name}", "",
               "| 추적기 | 유형 | 교차 | 가림 | 화면 경계 | 기타 | 합계 |", "|---|---|---:|---:|---:|---:|---:|"]
    all_events = []
    for name in args.trackers:
        events = []
        result_dir = output / name / "mot"
        evaluate_config(tracker_config(name), sequences, cached, result_dir, events)
        by_name = {s.name: s for s in sequences}
        gts = {s.name: load_ground_truth(s.gt_path)[0] for s in sequences}
        for event in events:
            sequence = by_name[event["sequence"]]
            event["category"], event["overlap"], event["visibility"] = classify(event, gts[sequence.name], sequence)
            event["tracker"] = name
        for kind in ("IDSW", "FRAG"):
            counts = Counter(e["category"] for e in events if e["type"] == kind)
            summary.append(f"| {name} | {kind} | {counts['교차']} | {counts['가림']} | "
                           f"{counts['화면 경계']} | {counts['기타']} | {sum(counts.values())} |")
        # 겹침이 큰(=분석 가치가 높은) ID 변경 사례부터 이미지로 저장합니다.
        cases = sorted([e for e in events if e["type"] == "IDSW"], key=lambda e: -e["overlap"])
        for number, event in enumerate(cases[:args.images]):
            sequence = by_name[event["sequence"]]
            draw_case(sequence, event, gts[sequence.name], result_dir / f"{sequence.name}.txt",
                      output / name / f"{number:02d}_{sequence.name}_{event['frame']:06d}.jpg")
        all_events += events

    output.mkdir(parents=True, exist_ok=True)
    with (output / "events.csv").open("w", encoding="utf-8-sig", newline="") as target:
        fields = ["tracker", "sequence", "type", "category", "frame", "gt_id", "old_id", "new_id",
                  "overlap", "visibility"]
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_events)
    summary += ["", "IDSW: 같은 사람의 추적 ID가 바뀐 횟수, FRAG: 추적이 끊겼다가 다시 이어진 횟수.",
                "교차 = 다른 사람과 IoU ≥ 0.3, 가림 = visibility < 0.5, 화면 경계 = 가장자리 3% 이내."]
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"\n결과 폴더: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("sequences", nargs="+")
    parser.add_argument("--trackers", nargs="+", default=["baseline", "improved"])
    parser.add_argument("--name", default="test")
    parser.add_argument("--images", type=int, default=10, help="추적기별 저장할 ID 변경 사례 이미지 수")
    parser.add_argument("--weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args())
