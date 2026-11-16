"""동일 초기 모델·데이터·시드로 학습률별 재학습 후 val 지표와 같은 영상의 추적 결과를 비교합니다."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from compare_camouflage_video import diagnose, sha256
from extensions.camouflage import CLASS_NAME, validate_nms
import run_camouflage
import train_camouflage


def prepare(args):
    rates = list(args.lr)
    if (not 2 <= len(rates) <= 6 or len(set(rates)) != len(rates)
            or any(not math.isfinite(v) or not 0 < v <= 1 for v in rates)):
        raise ValueError("서로 다른 학습률 2~6개를 0 초과 1 이하로 지정하세요.")
    if not 1 <= args.epochs <= 1000 or args.batch < 1 or args.imgsz < 32:
        raise ValueError("에폭은 1~1000, batch는 양수, imgsz는 32 이상이어야 합니다.")
    validate_nms(args.nms_iou)
    if args.frames is not None and args.frames < 1:
        raise ValueError("비교 영상 프레임 수는 양수여야 합니다.")
    if args.output.exists():
        raise FileExistsError("새 결과 폴더를 지정하세요. 기존 모델과 결과는 덮어쓰지 않습니다.")
    if not args.weights.is_file() or not args.video.is_file() or not args.data.is_file():
        raise FileNotFoundError("초기 가중치·원본 영상·dataset.yaml 경로를 확인하세요.")
    if args.device != "cpu":
        import torch
        if args.device != "0" or not torch.cuda.is_available():
            raise ValueError("GPU 0을 사용할 수 없습니다. CPU를 선택하거나 CUDA 환경을 준비하세요.")
    root = args.data.resolve().parent
    manifest_path = root / "splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("classes") != [CLASS_NAME]:
        raise ValueError("보호색 단일 클래스 데이터가 필요합니다.")
    # 이전 Colab 절대 경로 대신 이 PC의 실제 데이터 위치로 실행 YAML을 만듭니다.
    fingerprint, split_hashes, seen = hashlib.sha256(), {}, set()
    for split in ("train", "val"):
        records = [row for row in manifest["records"] if row["split"] == split]
        if not records or len(records) != manifest["counts"][split]["images"]:
            raise ValueError(f"{split} 전체 이미지와 명세가 필요합니다. 검증 데이터만으로는 학습할 수 없습니다.")
        if not any(row["boxes"] > 0 for row in records):
            raise ValueError(f"{split}에 정답 객체가 없습니다.")
        for index, row in enumerate(records, 1):
            relative = Path(row["image"])
            if relative.is_absolute() or len(relative.parts) != 3 or relative.parts[:2] != ("images", split):
                raise ValueError("학습·검증 이미지 경로 오류입니다.")
            image = (root / relative).resolve()
            label = (root / "labels" / split / relative.with_suffix(".txt").name).resolve()
            if not image.is_relative_to(root) or not label.is_relative_to(root) or image in seen:
                raise ValueError("중복 이미지 경로 또는 데이터 폴더 밖 경로입니다.")
            seen.add(image)
            with Image.open(image):
                pass
            label_bytes = label.read_bytes()
            lines = label_bytes.decode("utf-8").splitlines()
            if len(lines) != row["boxes"]:
                raise ValueError(f"정답 개수가 명세와 다릅니다: {label}")
            for line in lines:
                values = list(map(float, line.split()))
                if len(values) != 5 or not all(math.isfinite(v) for v in values):
                    raise ValueError(f"YOLO 정답 형식 오류: {label}")
                cls, x, y, w, h = values
                if (cls != 0 or not 0 < w <= 1 or not 0 < h <= 1 or
                        min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6):
                    raise ValueError(f"YOLO 정답 범위 오류: {label}")
            digest = sha256(image)
            if split_hashes.setdefault(digest, split) != split:
                raise ValueError("학습·검증 이미지 중복입니다. 데이터 검사를 먼저 수행하세요.")
            fingerprint.update(json.dumps([relative.as_posix(), digest, hashlib.sha256(label_bytes).hexdigest()]).encode())
            if index % 500 == 0 or index == len(records):
                print(f"데이터 확인 {split}: {index}/{len(records)}", flush=True)
    return dict(root=root, manifest=manifest, dataset_sha256=fingerprint.hexdigest(),
                weights_sha256=sha256(args.weights), video_sha256=sha256(args.video))


def write_report(output, report):
    target = output / "comparison.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def compare(args):
    info = prepare(args)
    output = args.output.resolve()
    output.mkdir(parents=True)
    config = output / "dataset_config"
    config.mkdir()
    data = config / "dataset.yaml"
    data.write_text(json.dumps(dict(path=str(info["root"]), train="images/train", val="images/val",
                                   names={0: CLASS_NAME})), encoding="utf-8")
    (config / "splits.json").write_text(json.dumps(info["manifest"]), encoding="utf-8")
    report = dict(kind="camouflage_learning_rates", status="running", candidates=[],
                  dataset_sha256=info["dataset_sha256"], initial_weights_sha256=info["weights_sha256"],
                  video_sha256=info["video_sha256"], epochs=args.epochs, seed=args.seed,
                  imgsz=args.imgsz, batch=args.batch, nms_iou=args.nms_iou, device=args.device,
                  selection="val mAP50-95 최대, 동률이면 입력 순서", tracking_evaluation="미실행",
                  protocol="후보마다 같은 초기 가중치로 새 학습. test 평가 안 함. patience=0으로 조기 종료 끔.")
    write_report(output, report)
    try:
        for index, rate in enumerate(args.lr):
            print(f"학습 후보 {index+1}/{len(args.lr)}: 초기 학습률={rate}", flush=True)
            if sha256(args.weights) != info["weights_sha256"]:
                raise ValueError("실험 도중 초기 가중치가 변경됐습니다.")
            candidate = output / f"candidate_{index:02d}"
            candidate.mkdir()
            training = train_camouflage.parser().parse_args([
                "--data", str(data), "--weights", str(args.weights.resolve()),
                "--epochs", str(args.epochs), "--imgsz", str(args.imgsz), "--batch", str(args.batch),
                "--lr0", str(rate), "--seed", str(args.seed), "--device", args.device,
                "--workers", "0", "--patience", "0", "--val-only", "--runs", str(candidate/"training")])
            metadata = train_camouflage.train(training)
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in metadata["scores"]["val"].values()):
                raise ValueError("val 지표가 유효한 0~1 범위가 아닙니다.")
            weights = train_camouflage.install_model(metadata["best"], metadata, candidate/"model")
            video_options = run_camouflage.parser().parse_args([
                "--video", str(args.video.resolve()), "--weights", str(weights),
                "--device", args.device, "--nms-iou", str(args.nms_iou), "--output", str(candidate/"video"),
                *(["--frames", str(args.frames)] if args.frames is not None else [])])
            run_camouflage.run(video_options)
            report["candidates"].append(dict(index=index, lr0=rate, val=metadata["scores"]["val"],
                                              weights=str(weights.relative_to(output)),
                                              video=str((candidate/"video/tracked.mp4").relative_to(output)),
                                              diagnostics=diagnose(candidate/"video/tracks.csv")))
            write_report(output, report)
        if sha256(args.video) != info["video_sha256"] or sha256(args.weights) != info["weights_sha256"]:
            raise ValueError("실험 도중 원본 영상 또는 초기 가중치가 변경됐습니다.")
        report["best_candidate"] = max(report["candidates"], key=lambda row: row["val"]["mAP50-95"])["index"]
        report["status"] = "completed"
        with (output/"comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["candidate", "lr0", "val_mAP50", "val_mAP50-95"])
            writer.writeheader()
            writer.writerows(dict(candidate=row["index"], lr0=row["lr0"], val_mAP50=row["val"]["mAP50"],
                                  **{"val_mAP50-95": row["val"]["mAP50-95"]}) for row in report["candidates"])
        write_report(output, report)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        write_report(output, report)
        raise
    print(f"학습률 비교 완료: {output/'comparison.json'}", flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True, help="학습 정답이 아닌 참고 비교용 원본 영상")
    p.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent/"yolo11n.pt")
    p.add_argument("--lr", type=float, nargs="+", default=[0.0001, 0.0005])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nms-iou", type=float, default=0.3)
    p.add_argument("--frames", type=int)
    p.add_argument("--device", choices=["cpu", "0"], default="cpu")
    p.add_argument("--output", type=Path, required=True)
    return p


if __name__ == "__main__":
    compare(parser().parse_args())
