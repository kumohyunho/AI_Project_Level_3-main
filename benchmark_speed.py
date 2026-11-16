"""처리 속도 측정과 속도 영향 요인 분석 (요구 사항 8).

같은 영상 구간에서 요인을 하나씩 바꿔 프레임당 평균 처리 시간(검출/추적/모자이크)과 FPS를 측정합니다.

* 입력 해상도(imgsz): 검출 시간에 가장 큰 영향을 줍니다.
* 검출 가중치: 모델 크기가 같으면(YOLO11n) 속도는 비슷하고 정확도만 달라집니다.
* 추적기: 개선판은 외형 특징 계산이 추가됩니다.
* 얼굴 모자이크: 얼굴 검출기를 한 번 더 실행합니다.
* 장면의 사람 수: 프레임별 검출 수와 처리 시간의 상관계수로 확인합니다.

예)
    python benchmark_speed.py sample.mp4 --frames 150
결과: results/speed.json, results/speed.md (GUI '성능 비교' 탭에도 표시)
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np

from pipeline import MODEL_DIR, ROOT, PipelineConfig, TrackingPipeline, default_weights


def read_frames(source, count):
    import cv2

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"영상을 읽을 수 없습니다: {source}")
    return frames, fps


def measure(config, frames, fps, warmup=5):
    pipeline = TrackingPipeline(config, fps)
    timings, counts = [], []
    for index, frame in enumerate(frames):
        rows, _, timing = pipeline.process(frame.copy())
        if index >= warmup:  # 첫 몇 프레임은 모델 초기화 비용이 섞여 제외합니다.
            timings.append(timing)
            counts.append(len(rows))
    stage = {key: float(np.mean([t[key] for t in timings])) for key in ("detect", "track", "mosaic", "total")}
    totals = np.array([t["total"] for t in timings])
    correlation = float(np.corrcoef(counts, totals)[0, 1]) if np.std(counts) > 0 else 0.0
    return dict(stage, fps=1000 / stage["total"], p95_ms=float(np.percentile(totals, 95)),
                mean_objects=float(np.mean(counts)), objects_time_corr=correlation)


def main(args):
    frames, fps = read_frames(args.video, args.frames)
    weights = args.weights or default_weights()
    base = dict(weights=weights, tracker="baseline", imgsz=640, device=args.device, mosaic=False)
    experiments = [("기준 (640, baseline, 모자이크 끔)", {})]
    experiments += [(f"입력 크기 {size}", dict(imgsz=size)) for size in (320, 480, 960)]
    experiments += [("추적기 improved", dict(tracker="improved"))]
    if (MODEL_DIR / "face.pt").is_file():
        experiments += [("얼굴 모자이크 켬", dict(mosaic=True))]
    for extra in args.compare_weights or []:
        experiments += [(f"가중치 {Path(extra).name}", dict(weights=extra))]

    results = []
    for label, change in experiments:
        config = PipelineConfig(**dict(base, **change))
        result = measure(config, frames, fps)
        result.update(label=label, weights=Path(config.weights).name, imgsz=config.imgsz,
                      tracker=config.tracker, mosaic_on=config.mosaic)
        results.append(result)
        print(f"{label:<28} 검출 {result['detect']:6.1f}ms  추적 {result['track']:5.1f}ms  "
              f"모자이크 {result['mosaic']:5.1f}ms  → {result['fps']:5.1f} FPS")

    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    meta = dict(video=str(args.video), frames=len(frames), device=args.device,
                cpu=platform.processor() or platform.machine(), python=platform.python_version())
    (output / "speed.json").write_text(json.dumps(dict(meta=meta, results=results), ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
    lines = [f"측정 환경: {meta['cpu']}, device={args.device}, {len(frames)}프레임 ({Path(args.video).name})", "",
             "| 조건 | 검출 ms | 추적 ms | 모자이크 ms | 전체 ms | p95 ms | FPS | 평균 객체 수 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['label']} | {r['detect']:.1f} | {r['track']:.2f} | {r['mosaic']:.1f} | "
                     f"{r['total']:.1f} | {r['p95_ms']:.1f} | {r['fps']:.1f} | {r['mean_objects']:.1f} |")
    base_corr = results[0]["objects_time_corr"]
    lines += ["", f"기준 조건에서 프레임별 객체 수와 처리 시간의 상관계수: {base_corr:.2f}"]
    (output / "speed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--weights")
    parser.add_argument("--compare-weights", nargs="*", help="함께 비교할 다른 가중치")
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args())
