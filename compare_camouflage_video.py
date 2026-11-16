"""같은 영상·모델·추적기로 NMS 전후 영상과 진단표를 생성합니다. 정답 없는 정확도 평가는 아닙니다."""
import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

from compare_camouflage_detection import overlap
from extensions.camouflage import validate_nms
from results_io import load_results
import run_camouflage


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def diagnose(path):
    frames, metadata = load_results(path)
    total = metadata["frames"]
    ids = set()
    multiple = pairs = overlapping_frames = 0
    examples = []
    for number, rows in frames.items():
        ids.update(row["id"] for row in rows)
        multiple += int(len(rows) > 1)
        count = sum(float(overlap(a["box"], [b["box"]])[0]) >= 0.5
                    for a, b in combinations(rows, 2))
        pairs += count
        overlapping_frames += int(count > 0)
        if count and len(examples) < 20:
            examples.append(number)
    return dict(frames=total, unique_ids=len(ids), empty_frames=total-len(frames),
                multiple_tracks_frames=multiple, overlap_frames=overlapping_frames,
                overlap_pairs=pairs, example_overlap_frames=examples)


def compare(args):
    validate_nms(args.before_nms)
    validate_nms(args.after_nms)
    if args.frames is not None and args.frames < 1:
        raise ValueError("처리 프레임 수는 양수여야 합니다.")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("기존 결과를 보존합니다. 새 출력 폴더를 지정하세요.")
    video_hash, weights_hash = sha256(args.video), sha256(args.weights)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    with TemporaryDirectory(dir=output.parent) as folder:
        staged = Path(folder) / "comparison"
        staged.mkdir()
        for name, nms in (("before", args.before_nms), ("after", args.after_nms)):
            print(f"{name}: NMS IoU={nms} 영상 처리 시작", flush=True)
            options = run_camouflage.parser().parse_args([
                "--video", str(args.video), "--weights", str(args.weights),
                "--device", args.device, "--output", str(staged/name), "--nms-iou", str(nms),
                *(["--frames", str(args.frames)] if args.frames is not None else [])])
            metadata = run_camouflage.run(options)
            results[name] = dict(nms_iou=nms, diagnostics=diagnose(staged/name/"tracks.csv"),
                                 tracker=metadata["tracker"], detection=metadata["detection"])
        if (results["before"]["tracker"] != results["after"]["tracker"] or
                results["before"]["diagnostics"]["frames"] != results["after"]["diagnostics"]["frames"]):
            raise ValueError("비교 실행의 추적 설정 또는 프레임 수가 다릅니다.")
        if video_hash != sha256(args.video) or weights_hash != sha256(args.weights):
            raise ValueError("비교 도중 입력 영상 또는 가중치가 변경됐습니다.")
        report = dict(video_sha256=video_hash, weights_sha256=weights_hash, **results,
                      tracking_evaluation="미실행: 정답 ID를 사용하지 않은 진단",
                      limitations=["겹치는 트랙은 실제 두 동물일 수도 있어 중복 확정값이 아닙니다.",
                                   "겹침·ID 수 감소는 미검출 증가로도 발생합니다. 두 영상을 함께 검토하세요.",
                                   "NMS만 변경합니다. 신뢰도 0.2의 검출 F1을 재현하는 실험이 아닙니다.",
                                   "짧은 영상의 처리 시간은 초기 추론 영향을 받으므로 속도 개선 근거로 쓰지 마세요."])
        (staged/"comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        columns = ["setting", "nms_iou", "frames", "unique_ids", "empty_frames",
                   "multiple_tracks_frames", "overlap_frames", "overlap_pairs"]
        with (staged/"comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for name, result in results.items():
                writer.writerow(dict(setting=name, nms_iou=result["nms_iou"],
                                     **{key: result["diagnostics"][key] for key in columns[2:]}))
        staged.rename(output)
    print(f"비교 결과: {output}\n실제 정확도 개선 여부는 아직 검증하지 않았습니다.")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=Path("models/camouflage.pt"))
    p.add_argument("--before-nms", type=float, default=0.7)
    p.add_argument("--after-nms", type=float, default=0.3)
    p.add_argument("--frames", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    return p


if __name__ == "__main__":
    compare(parser().parse_args())
