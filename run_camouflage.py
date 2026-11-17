"""새로운 영상·카메라의 보호색 동물 추적 → 박스·ID·궤적 영상과 CSV.

GUI 없이 실행할 때 사용합니다. 한국어 화면은 main.py의 보호색 동물 확장을 사용하세요.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from extensions.camouflage import CamouflageExtension, read_tracker_config, validate_nms, validate_tracker
from results_io import ResultWriter

ROOT = Path(__file__).resolve().parent


def make_extension(weights, device="cpu", fps=30.0, tracker=None, nms_iou=None):
    class Selected(CamouflageExtension):
        @classmethod
        def weights_path(cls):
            return Path(weights).resolve()

    extension = Selected(device=device, fps=fps, nms_iou=nms_iou)
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
    nms_iou = getattr(args, "nms_iou", None)
    if nms_iou is not None:
        validate_nms(nms_iou)
    source = args.camera if args.camera is not None else str(args.video)
    if args.camera is None and not Path(source).is_file():
        raise FileNotFoundError(f"영상이 없습니다: {source}")
    capture = cv2.VideoCapture(source)
    writer = extension = diagnostic = diagnostic_video = None
    results = ResultWriter()
    frames, elapsed = 0, 0.0
    try:
        if not capture.isOpened():
            raise ValueError("영상 또는 카메라를 열 수 없습니다.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and 0 < fps < 240 else 30.0
        config = read_tracker_config(args.tracker_config, args.weights) if args.tracker_config else None
        extension = make_extension(args.weights, args.device, fps, config, nms_iou=nms_iou)
        output.mkdir(parents=True)
        if getattr(args, "diagnostics", False):
            from camouflage_diagnostics import DiagnosticRecorder, draw_detections, file_sha256, label_panel

            reported_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) if args.camera is None else 0
            diagnostic = DiagnosticRecorder(output, dict(
                weights_sha256=file_sha256(args.weights),
                model_metadata_sha256=file_sha256(Path(args.weights).with_suffix(".json")),
                inference_file_sha256=(file_sha256(Path(args.weights).with_suffix(".inference.json"))
                                       if Path(args.weights).with_suffix(".inference.json").is_file() else None),
                source_sha256=file_sha256(source) if args.camera is None else None,
                input_fps=fps, python=sys.version.split()[0], opencv=cv2.__version__,
                ultralytics=getattr(sys.modules.get("ultralytics"), "__version__", None),
                torch=getattr(sys.modules.get("torch"), "__version__", None),
                device=args.device,
                frame_limit=args.frames, source_type="camera" if args.camera is not None else "video",
                container_reported_frames=int(reported_frames) if math.isfinite(reported_frames) and reported_frames > 0 else None,
                detection=extension.inference_config, tracker=extension.tracker_config.to_dict()))
            extension.enable_diagnostics(diagnostic)
        while args.frames is None or frames < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("MP4 출력 코덱을 열 수 없습니다.")
                if diagnostic is not None:
                    diagnostic_video = cv2.VideoWriter(str(output/"diagnostic.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w*2, h))
                    if not diagnostic_video.isOpened():
                        raise RuntimeError("진단 영상 출력 코덱을 열 수 없습니다.")
            frames += 1
            started = perf_counter()
            rows = extension.process(frame, frames)
            elapsed += perf_counter() - started
            results.add(frames, (frames - 1) * 1000 / fps, rows)
            if diagnostic is not None:
                detection_panel = draw_detections(frame, extension.byte_tracker.trace)
            draw_tracks(frame, rows)
            writer.write(frame)
            if diagnostic_video is not None:
                tracking_panel = label_panel(frame.copy(), f"FRAME {frames} | VISIBLE TRACKS")
                diagnostic_video.write(np.hstack((detection_panel, tracking_panel)))
            if diagnostic is not None and frames % 25 == 0:
                print(f"진단 기록: {frames}프레임", flush=True)
        if not frames:
            raise ValueError("읽은 영상 프레임이 없습니다.")
        meta = dict(source=str(Path(source).resolve()) if args.camera is None else str(source),
                    fps=fps, frames=frames, pipeline=dict(extensions=["camouflage"], weights=str(args.weights)),
                    tracker=extension.tracker_config.to_dict(),
                    detection=getattr(extension, "inference_config", None),
                    processing_ms=elapsed * 1000 / frames, processing_fps=frames / max(elapsed, 1e-9),
                    timing_scope="모델 초기화·파일 입출력·화면 그리기를 제외한 검출+추적 평균",
                    note="검출된 트랙만 저장. 완전 가림 중의 예측은 실제 검출 결과로 저장하지 않음.")
        if diagnostic is not None:
            meta["diagnostics"] = dict(summary="diagnostic_summary.json", frames="diagnostic_frames.jsonl",
                                       events="diagnostic_events.csv", report="diagnostic_report.html", video="diagnostic.mp4")
            meta["timing_scope"] = "모델 초기화·영상 입출력·화면 그리기 제외 · 검출·추적·진단 분석·프레임 기록 포함"
        writer.release()
        writer = None
        if diagnostic_video is not None:
            diagnostic_video.release()
            diagnostic_video = None
        results.save(output / "tracks.csv", meta)
        if diagnostic is not None:
            if diagnostic.frames != frames:
                raise RuntimeError("처리 프레임 수와 진단 기록 수가 다릅니다.")
            diagnostic.finish("completed", dict(processed_frames=frames))
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if diagnostic_video is not None:
            diagnostic_video.release()
        if diagnostic is not None and not diagnostic.closed:
            diagnostic.finish("failed", dict(error=str(sys.exc_info()[1] or "처리 미완료")))
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
    p.add_argument("--tracker-config", type=Path, help="검증 영상에서 고른 추적 설정 JSON")
    p.add_argument("--nms-iou", type=float, help="보호색 NMS 명시값. 생략 시 모델 옆 실행 설정 또는 기존 기본값")
    p.add_argument("--frames", type=int, help="동작 확인 시 처리 프레임 제한")
    p.add_argument("--diagnostics", action="store_true", help="프레임 연결 기록·의심 구간 요약·검출/추적 진단 영상 저장")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
