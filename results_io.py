"""추적 결과 파일 저장·불러오기 (요구 사항 10).

* CSV: ``frame, time_ms, id, class, x1, y1, x2, y2, score, status, direction, distance_px, speed_px_s``
  한 줄이 한 프레임의 한 객체입니다. 엑셀에서 바로 열 수 있도록 UTF-8 BOM으로 저장합니다.
* 같은 이름의 ``.json``: 원본 영상 경로, FPS, 모델·추적기 설정을 함께 저장해 결과를 재현할 수 있게 합니다.
* MOT txt: MOTChallenge/TrackEval 형식 ``frame,id,left,top,width,height,conf,-1,-1,-1``.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

COLUMNS = ["frame", "time_ms", "id", "class", "x1", "y1", "x2", "y2", "score", "status",
           "direction", "distance_px", "speed_px_s"]


class ResultWriter:
    def __init__(self):
        self.rows = []

    def add(self, frame, time_ms, tracks):
        for t in tracks:
            x1, y1, x2, y2 = t.box
            self.rows.append([frame, time_ms, t.track_id, t.cls_name, f"{x1:.1f}", f"{y1:.1f}",
                              f"{x2:.1f}", f"{y2:.1f}", f"{t.score:.3f}", t.status, t.direction,
                              f"{t.distance:.1f}", f"{t.speed:.1f}"])

    def __len__(self):
        return len(self.rows)

    def save(self, path, metadata):
        path = Path(path)
        with path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(COLUMNS)
            writer.writerows(self.rows)
        path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_results(path):
    """CSV를 읽어 ``{frame: [row dict, ...]}``와 메타데이터를 돌려줍니다."""
    path = Path(path)
    frames = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"추적 결과 CSV에 열이 없습니다: {', '.join(sorted(missing))}")
        for row in reader:
            frames[int(row["frame"])].append({
                "id": int(row["id"]), "class": row["class"],
                "box": tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")),
                "score": float(row["score"]), "status": row["status"],
                "direction": row["direction"], "distance": float(row["distance_px"]),
                "speed": float(row["speed_px_s"]),
            })
    sidecar = path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    return dict(frames), metadata


def write_mot(path, rows):
    """``rows``: (frame, id, x1, y1, x2, y2, score) 목록."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        for frame, track_id, x1, y1, x2, y2, score in rows:
            target.write(f"{frame},{track_id},{x1 + 1:.2f},{y1 + 1:.2f},{x2 - x1:.2f},"
                         f"{y2 - y1:.2f},{score:.4f},-1,-1,-1\n")
