"""YOLO11n 검출기 Fine-tuning (요구 사항 6·10). ByteTrack 자체는 학습할 구성 요소가 없습니다.

| 종류  | 데이터                         | 설치 경로          | 용도                          |
|-------|--------------------------------|--------------------|-------------------------------|
| coco  | prepare_dataset.py coco (4클래스) | models/coco.pt   | 사람·개·고양이·새 검출          |
| crowd | prepare_dataset.py crowd       | models/person.pt   | 붐비는 장면의 사람 검출          |
| mot   | prepare_dataset.py mot         | models/person.pt   | 길거리 보행자 검출 (MOT 영상)    |
| face  | prepare_dataset.py face        | models/face.pt     | 얼굴 모자이크 (확장 기능)        |

학습이 끝나면 **test 분할**로 최종 성능을 측정하고, ``--install``을 주면 best.pt를 위 경로로 복사하면서
학습 설정과 성능을 같은 이름의 ``.json``으로 함께 저장합니다.

예)
    python train_models.py mot --data datasets/mot/dataset.yaml --weights yolo11n.pt --epochs 30 --install
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

INSTALL = {"coco": "coco.pt", "crowd": "person.pt", "mot": "person.pt", "face": "face.pt"}


def train(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    options = {
        "data": args.data, "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "device": args.device, "workers": args.workers, "project": str(Path(args.runs).resolve()),
        "name": args.kind, "seed": 42, "optimizer": args.optimizer, "lr0": args.lr0,
        "freeze": args.freeze, "mosaic": args.mosaic, "patience": args.patience,
    }
    model.train(**options)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("학습에서 best.pt가 생성되지 않았습니다.")

    tuned = YOLO(str(best))
    scores = {}
    for split in ("val", "test"):
        try:
            result = tuned.val(data=args.data, split=split, imgsz=args.imgsz, batch=args.batch,
                               device=args.device, workers=args.workers, verbose=False, plots=False)
            scores[split] = {"mAP50": float(result.box.map50), "mAP50-95": float(result.box.map)}
        except Exception as error:  # dataset.yaml에 test가 없을 수 있습니다.
            scores[split] = {"error": str(error)}
    print(f"학습 완료: {best.resolve()}")
    print(json.dumps(scores, ensure_ascii=False, indent=2))

    metadata = dict(kind=args.kind, base_weights=args.weights, created=datetime.now().isoformat(timespec="seconds"),
                    train_options=options, scores=scores, best=str(best.resolve()))
    best.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.install:
        target = Path(__file__).resolve().parent / "models" / INSTALL[args.kind]
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            backup = target.with_name(f"{target.stem}_{datetime.now():%Y%m%d_%H%M%S}.pt")
            shutil.move(target, backup)
            print(f"기존 가중치 백업: {backup}")
        shutil.copyfile(best, target)
        target.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        print(f"설치: {target} (+ 학습 설정 {target.with_suffix('.json').name})")
    else:
        print(f"GUI에서 사용하려면 --install을 주거나 models/{INSTALL[args.kind]}로 복사하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("kind", choices=list(INSTALL))
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--freeze", type=int, default=10, help="고정할 앞쪽 레이어 수 (적은 데이터에서 과적합 방지)")
    parser.add_argument("--mosaic", type=float, default=1.0, help="모자이크 증강 확률 (0이면 끔)")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu", help="cpu 또는 CUDA 장치 번호")
    parser.add_argument("--workers", type=int, default=0, help="데이터 로딩 프로세스 수 (Windows는 0 권장, Colab·Linux GPU는 4 정도)")
    parser.add_argument("--runs", default="training_runs")
    parser.add_argument("--install", action="store_true", help="best.pt를 models/에 설치")
    train(parser.parse_args())
