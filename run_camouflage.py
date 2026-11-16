"""새로운 영상·카메라의 보호색 동물 추적 → 박스·ID·궤적 영상과 CSV.

GUI 없이 실행할 때 사용합니다. 한국어 화면은 main.py의 보호색 동물 확장을 사용하세요.
"""
import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from extensions.camouflage import CamouflageExtension, validate_tracker
from results_io import ResultWriter

ROOT = Path(__file__).resolve().parent


def make_extension(weights, device="cpu", fps=30.0, tracker=None):
    class Selected(CamouflageExtension):
        @classmethod
        def weights_path(cls):
            return Path(weights).resolve()

    extension = Selected(device=device, fps=fps)
    if tracker is not None:
        from tracker import ByteTracker

        extension.tracker_config = validate_tracker(tracker)
        extension.byte_tracker = ByteTracker(extension.tracker_config, fps)
    return extension


def draw_tracks(frame, rows):
    import cv2

    for row in rows:
        color = tuple(int(v) for v in np.random.default_rng(row.track_id).integers(70, 256, 3))
        x1, y1, x2, y2 = map(int, row.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"CAM {row.track_id} {row.score:.2f}", (max(0, x1), max(15, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if len(row.trail) > 1:
            cv2.polylines(frame, [np.array(row.trail, dtype=np.int32)], False, color, 2)


def run(args):
    import cv2

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"기존 결과를 보존합니다. 다른 출력 폴더를 지정하세요: {output}")
    source = args.camera if args.camera is not None else str(args.video)
    if args.camera is None and not Path(source).is_file():
        raise FileNotFoundError(f"영상이 없습니다: {source}")
    capture = cv2.VideoCapture(source)
    writer = extension = None
    results = ResultWriter()
    frames, elapsed = 0, 0.0
    try:
        if not capture.isOpened():
            raise ValueError("영상 또는 카메라를 열 수 없습니다.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and 0 < fps < 240 else 30.0
        extension = make_extension(args.weights, args.device, fps)
        output.mkdir(parents=True)
        while args.frames is None or frames < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("MP4 출력 코덱을 열 수 없습니다.")
            frames += 1
            started = perf_counter()
            rows = extension.process(frame, frames)
            elapsed += perf_counter() - started
            results.add(frames, (frames - 1) * 1000 / fps, rows)
            draw_tracks(frame, rows)
            writer.write(frame)
        if not frames:
            raise ValueError("읽은 영상 프레임이 없습니다.")
        meta = dict(source=str(Path(source).resolve()) if args.camera is None else str(source),
                    fps=fps, frames=frames, pipeline=dict(extensions=["camouflage"], weights=str(args.weights)),
                    tracker=extension.tracker_config.to_dict(),
                    processing_ms=elapsed * 1000 / frames, processing_fps=frames / max(elapsed, 1e-9),
                    timing_scope="모델 초기화·파일 입출력·화면 그리기를 제외한 검출+추적 평균",
                    note="검출된 트랙만 저장. 완전 가림 중의 예측은 실제 검출 결과로 저장하지 않음.")
        results.save(output / "tracks.csv", meta)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if extension is not None:
            extension.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--camera", type=int)
    p.add_argument("--weights", type=Path, default=ROOT / "models" / "camouflage.pt")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "camouflage" / "video")
    p.add_argument("--device", default="cpu")
    p.add_argument("--frames", type=int, help="동작 확인 시 처리 프레임 제한")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
