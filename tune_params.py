"""추적 파라미터 격자 탐색 (요구 사항 6).

검출 임계값(high_thresh), 연관성 임계값(match_thresh), 추적 유지 시간(track_buffer)을 바꿔 가며
**검증(val) 구간**에서 HOTA가 가장 높은 조합을 고르고 ``configs/<이름>.json``으로 저장합니다.
검출은 캐시를 재사용하므로 조합 하나당 추적기만 다시 돌립니다.
고른 설정은 **테스트(test) 구간**에서 ``run_mot.py``로 최종 확인하세요 (val에서 고르고 test에서 보고).

예)
    python tune_params.py datasets/mot/tracking/val --base improved --name tuned
    python run_mot.py datasets/mot/tracking/test --trackers baseline improved tuned --name test
"""
import argparse
import csv
import itertools
import json
from dataclasses import replace
from datetime import datetime

from mot_data import find_sequences, load_detections
from pipeline import CONFIG_DIR, ROOT, default_weights, tracker_config
from run_mot import evaluate_config

GRID = {
    "high_thresh": [0.4, 0.5, 0.6],
    "match_thresh": [0.7, 0.8, 0.9],
    "track_buffer": [30, 60, 90],
}


def parse_grid(values):
    grid = dict(GRID)
    for item in values or []:
        key, _, numbers = item.partition("=")
        if key not in grid:
            raise ValueError(f"탐색할 수 없는 항목: {key} (가능: {', '.join(GRID)})")
        cast = int if key == "track_buffer" else float
        grid[key] = [cast(v) for v in numbers.split(",")]
    return grid


def main(args):
    sequences = find_sequences(args.sequences)
    weights = args.weights or default_weights()
    cached = {s.name: load_detections(s, weights, args.imgsz, args.device) for s in sequences}
    base = tracker_config(args.base)
    grid = parse_grid(args.grid)
    keys = list(grid)
    rows, best = [], None
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        # 새 트랙은 1차 연결 임계값보다 조금 높은 점수에서만 만들도록 함께 조정합니다.
        config = replace(base, **params, new_track_thresh=params.get("high_thresh", base.high_thresh) + 0.1)
        metrics, _ = evaluate_config(config, sequences, cached)
        row = dict(params, **{k: round(metrics[k], 3) for k in ("HOTA", "IDF1", "MOTA")},
                   IDSW=metrics["IDSW"])
        rows.append(row)
        print(row)
        if best is None or metrics[args.metric] > best[0][args.metric]:
            best = (metrics, config, params)

    table = ROOT / "results" / f"tuning_{args.name}.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: -r[args.metric]))

    metrics, config, params = best
    CONFIG_DIR.mkdir(exist_ok=True)
    output = CONFIG_DIR / f"{args.name}.json"
    output.write_text(json.dumps({
        "description": f"{args.base} 기반, val {args.metric} 최대 조합",
        "created": datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights), "sequences": [s.name for s in sequences],
        "best_params": params, "val_metrics": {k: metrics[k] for k in ("HOTA", "IDF1", "MOTA", "IDSW")},
        "tracker": config.to_dict(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n최적 조합 {params} → {args.metric} {metrics[args.metric]:.2f}")
    print(f"설정: {output}\n전체 표: {table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("sequences", nargs="+")
    parser.add_argument("--base", default="improved", help="시작 설정 (baseline / improved)")
    parser.add_argument("--name", default="tuned", help="저장할 설정 이름 → configs/<name>.json")
    parser.add_argument("--metric", default="HOTA", choices=["HOTA", "IDF1", "MOTA"])
    parser.add_argument("--grid", nargs="*", help="예: high_thresh=0.4,0.5 track_buffer=30,60")
    parser.add_argument("--weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args())
