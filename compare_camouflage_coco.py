"""보호색 전용 모델과 COCO 사전학습 YOLO11n을 같은 정답 영상에서 비교하는 대조 실험입니다.

보호색 모델이 서로 붙은 동물 여러 마리를 박스 하나로 묶어 검출할 때, 그 원인이 보호색 학습 정답(이미지당 박스 1개)에만
있는지 확인합니다. 개체마다 박스가 그려진 COCO로 학습한 기본 YOLO11n을 **보호색 평가와 같은 추적 설정**
(``models/camouflage.json``의 ``tracker``)으로 실행하고, ``evaluate_camouflage.py``와 같은 정답·같은 평가 프레임으로 채점합니다.

함께 넘긴 예측 파일(``--compare``)까지 포함해, 모델마다 "박스가 동물을 묶는 정도"를 셉니다.
* 프레임당 예측 수와 정답 수
* 예측 박스 하나가 정답 박스 중심을 2개 이상 품은 비율 (묶음 검출)
* 예측 박스 평균 면적 / 정답 박스 평균 면적

사용 예 (시퀀스 폴더는 ``evaluate_camouflage.py``와 같은 MOT 형식: img1/, gt/gt.txt, seqinfo.ini, annotations.json):

    python compare_camouflage_coco.py datasets/camouflage/tracking/reference/dogs_b675c5a6 \\
        --classes dog horse sheep cow cat bear --name coco_animals \\
        --compare 보호색=results/camouflage_coco/dogs_b675c5a6/camouflage_cpu/dogs_b675c5a6.txt
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_mot(path):
    """MOT txt(frame,id,left,top,width,height,...) → {frame: [(x1, y1, x2, y2), ...]}"""
    frames = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").split():
        values = line.split(",")
        frame = int(float(values[0]))
        left, top, width, height = map(float, values[2:6])
        frames[frame].append((left, top, left + width, top + height))
    return frames


def grouping_stats(truth, predicted, frames):
    """예측 박스가 정답(동물)을 얼마나 묶어 검출하는지 셉니다."""
    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    def holds(box, gt):
        cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
        return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]

    boxes = [(f, b) for f in frames for b in predicted.get(f, [])]
    gt_boxes = [g for f in frames for g in truth.get(f, [])]
    grouped = sum(1 for f, b in boxes if sum(holds(b, g) for g in truth.get(f, [])) >= 2)
    mean_gt = statistics.mean(area(g) for g in gt_boxes) if gt_boxes else 0.0
    return dict(frames=len(frames),
                truth_per_frame=len(gt_boxes) / max(1, len(frames)),
                predicted_per_frame=len(boxes) / max(1, len(frames)),
                predicted=len(boxes), grouped=grouped,
                grouped_ratio=grouped / len(boxes) if boxes else 0.0,
                area_ratio=(statistics.mean(area(b) for _, b in boxes) / mean_gt) if boxes and mean_gt else 0.0)


def run_coco(sequence, classes, tracker_config, weights, device, output):
    """COCO 가중치로 검출하고 보호색 평가와 같은 추적 설정으로 추적해 MOT txt를 씁니다. 반환: 프레임당 처리 ms."""
    from pipeline import Detector
    from results_io import write_mot
    from tracker import ByteTracker

    detector = Detector(str(weights), classes, 640, device)
    tracker = ByteTracker(tracker_config, sequence.fps)
    rows, times = [], []
    for number in sequence.frames:
        image = sequence.read(number)
        started = time.perf_counter()
        boxes, scores, _ = detector(image, tracker_config.low_thresh)
        # 여러 동물 클래스를 합칠 때 클래스가 바뀌어도 ID가 끊기지 않도록 같은 클래스로 추적합니다.
        tracks = tracker.update(boxes, scores, np.zeros(len(boxes), dtype=int), None,
                                (sequence.width, sequence.height))
        times.append((time.perf_counter() - started) * 1000)
        rows += [(number, t.track_id, *[float(v) for v in t.box], float(t.score)) for t in tracks]
    write_mot(output, rows)
    return float(np.mean(times)) if times else 0.0


def main(args):
    from evaluate_camouflage import load_evaluation_annotations
    from metrics import combine, evaluate_files
    from mot_data import Sequence
    from tracker import TrackerConfig

    sequence = Sequence(args.sequence)
    # evaluate_camouflage.py와 같은 검사: 정답 ID 확인 여부, 평가 프레임 목록
    evaluated = load_evaluation_annotations([sequence])[sequence.name]["evaluated_frames"]
    info = json.loads((ROOT / "models" / "camouflage.json").read_text(encoding="utf-8"))
    tracker_config = TrackerConfig(**info["tracker"])
    folder = ROOT / "results" / "camouflage_coco" / sequence.name / args.name
    folder.mkdir(parents=True, exist_ok=True)
    result = folder / f"{sequence.name}.txt"
    ms = run_coco(sequence, args.classes, tracker_config, ROOT / args.weights, args.device, result)
    metrics = combine([evaluate_files(sequence.gt_path, result, evaluated, [])])
    metrics = {k: float(v) for k, v in metrics.items()}

    truth = load_mot(sequence.gt_path)
    frames = list(evaluated)
    groups = {f"COCO ({', '.join(args.classes)})": grouping_stats(truth, load_mot(result), frames)}
    for item in args.compare or []:
        label, path = item.split("=", 1)
        groups[label] = grouping_stats(truth, load_mot(path), frames)

    summary = dict(sequence=sequence.name, weights=args.weights, classes=args.classes, device=args.device,
                   tracker=info["tracker"], ms_per_frame=ms, metrics=metrics, grouping=groups)
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequence", help="MOT 형식 평가 시퀀스 폴더 (annotations.json 포함)")
    parser.add_argument("--classes", nargs="+", default=["dog"], help="COCO 클래스 이름 (기본: dog)")
    parser.add_argument("--weights", default="yolo11n.pt", help="COCO 사전학습 가중치 (저장소 기준 경로)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="coco_dog", help="결과 폴더 이름 (results/camouflage_coco/<영상>/<이름>)")
    parser.add_argument("--compare", nargs="*", metavar="이름=예측.txt", help="묶음 검출 통계를 함께 낼 다른 예측 MOT 파일")
    main(parser.parse_args())
