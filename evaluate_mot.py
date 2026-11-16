r"""공식 TrackEval로 HOTA, MOTA, IDF1, IDSW를 교차 확인합니다 (선택).

run_mot.py는 같은 계산을 metrics.py로 수행합니다. 보고서에 공식 도구 수치가 필요할 때 사용하세요.
    python evaluate_mot.py --benchmark MOT17 --ground-truth datasets/mot/tracking/test \\
        --results results/mot/test --trackers baseline improved --sequences MOT17-02-FRCNN
"""
import argparse
from pathlib import Path
import sys


def evaluate(args):
    import numpy as np

    checkout = args.trackeval.resolve()
    if not (checkout / "trackeval" / "__init__.py").is_file():
        raise FileNotFoundError(f"TrackEval 저장소 경로를 확인하세요: {checkout}")
    sys.path.insert(0, str(checkout))
    # 일부 TrackEval 버전에서 쓰는 NumPy 별칭을 이 평가 프로세스에서 복원합니다.
    for name, scalar in (("float", float), ("int", int)):
        if name not in vars(np):
            setattr(np, name, scalar)
    import trackeval

    evaluator = trackeval.Evaluator({"PLOT_CURVES": False})
    dataset = trackeval.datasets.MotChallenge2DBox({
        "GT_FOLDER": str(args.ground_truth.resolve()),
        "TRACKERS_FOLDER": str(args.results.resolve()),
        "TRACKERS_TO_EVAL": args.trackers,
        "BENCHMARK": args.benchmark,
        "SPLIT_TO_EVAL": "train",
        "SKIP_SPLIT_FOL": True,
        "TRACKER_SUB_FOLDER": args.sub_folder,
        "SEQ_INFO": dict.fromkeys(args.sequences),
    })
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator.evaluate([dataset], metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trackeval", type=Path, default=Path("TrackEval"))
    parser.add_argument("--benchmark", choices=["MOT17", "MOT20"], required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--trackers", nargs="+", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--sub-folder", default="", help="추적기 폴더 안의 결과 하위 폴더 (run_mot.py 결과는 비워 둠)")
    evaluate(parser.parse_args())
