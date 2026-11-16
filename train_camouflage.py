"""보호색 동물 YOLO 추가 학습과 가중치·학습 설정 설치. docs/CAMOUFLAGE.md 참고."""
import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from extensions.camouflage import CLASS_NAME, default_tracker

ROOT = Path(__file__).resolve().parent


def install_model(weights, metadata, models):
    """학습 정보와 가중치를 함께 설치합니다. 기존 파일은 덮어쓰지 않습니다."""
    models = Path(models)
    models.mkdir(parents=True, exist_ok=True)
    target = models / "camouflage.pt"
    if target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError("기존 camouflage.pt/.json을 함께 백업한 뒤 다시 설치하세요.")
    if not 0 < Path(weights).stat().st_size < 100_000_000:
        raise ValueError("가중치는 비어 있지 않고 100MB 미만이어야 합니다.")
    with TemporaryDirectory(dir=models) as folder:
        staged = Path(folder) / "camouflage.pt"
        shutil.copyfile(weights, staged)
        payload = dict(metadata, weights_sha256=hashlib.sha256(staged.read_bytes()).hexdigest())
        config = staged.with_suffix(".json")
        config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        staged.rename(target)
        config.rename(target.with_suffix(".json"))
    return target


def train(args):
    # 데이터나 설치 경로에 문제가 있으면 긴 학습을 시작하기 전에 알립니다.
    data = Path(args.data).resolve()
    manifest_path = data.parent / "splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("classes") != [CLASS_NAME]:
        raise ValueError("camouflage_dataset.py로 만든 단일 클래스 데이터를 사용하세요.")
    splits = ("val",) if args.val_only else ("val", "test")
    if any(manifest["counts"][split]["images"] < 1 for split in ("train", *splits)):
        raise ValueError("학습·평가할 분할에 이미지가 필요합니다.")
    if not math.isfinite(args.lr0) or not 0 < args.lr0 <= 1:
        raise ValueError("초기 학습률은 0보다 크고 1 이하여야 합니다.")
    if args.install and any((ROOT / "models" / f"camouflage.{suffix}").exists() for suffix in ("pt", "json")):
        raise FileExistsError("models/camouflage.pt와 .json을 먼저 함께 백업하세요.")
    from ultralytics import YOLO

    options = dict(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                   device=args.device, workers=args.workers, seed=args.seed, optimizer="AdamW",
                   lr0=args.lr0, freeze=args.freeze, project=str(Path(args.runs).resolve()),
                   name="camouflage", patience=args.patience)
    model = YOLO(str(args.weights))
    model.train(**options)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("학습 결과 best.pt가 없습니다.")
    tuned = YOLO(str(best))
    if set(tuned.names.values()) != {CLASS_NAME}:
        raise ValueError("학습 가중치의 클래스가 보호색 동물 단일 클래스가 아닙니다.")
    scores = {}
    for split in splits:
        result = tuned.val(data=str(data), split=split, imgsz=args.imgsz, batch=args.batch,
                           device=args.device, workers=args.workers, verbose=False, plots=False)
        scores[split] = {"mAP50": float(result.box.map50), "mAP50-95": float(result.box.map)}
    metadata = dict(extension="camouflage", classes=[CLASS_NAME],
                    created=datetime.now(timezone.utc).isoformat(), base_weights=str(args.weights),
                    train_options=options, scores=scores, counts=manifest["counts"],
                    dataset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    tracker=default_tracker().to_dict(), tracker_status="시작값, 실영상 튜닝 미실행",
                    tracking_evaluation="미실행", best=str(best.resolve()))
    metadata["evaluation_splits"] = list(splits)
    best.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    installed = install_model(best, metadata, ROOT / "models") if args.install else None
    print(json.dumps(dict(best=str(best), installed=str(installed) if installed else None,
                          scores=scores), ensure_ascii=False, indent=2))
    return metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("datasets/camouflage/dataset.yaml"))
    p.add_argument("--weights", type=Path, default=ROOT / "yolo11n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--freeze", type=int, default=0)
    p.add_argument("--lr0", type=float, default=0.0005, help="AdamW 초기 학습률")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--val-only", action="store_true", help="모델 선택 실험에서는 test를 평가하지 않음")
    p.add_argument("--device", default="cpu", help="Colab GPU에서는 0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=Path, default=ROOT / "training_runs")
    p.add_argument("--install", action="store_true")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
