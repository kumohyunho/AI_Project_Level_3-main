"""프레임별 실제 연결 기록과 정답 없는 검토 후보 요약. IDSW·정확도를 추정하지 않습니다."""
import csv
import html
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from compare_camouflage_detection import overlap

EVENT_LABELS = {
    "detection_overlap": "검출 박스 겹침 후보",
    "track_overlap": "출력 트랙 겹침 후보",
    "empty_detections": "검출 없는 프레임",
    "detections_without_tracks": "검출은 있으나 확정 출력 없음 (확인 대기 포함)",
    "output_gap_started": "이전 프레임의 출력 ID가 사라짐",
    "tracker_reappearance": "추적기가 같은 ID로 다시 연결함",
    "id_replacement_candidate": "최근 사라진 ID 근처에 새 ID 생성 (IDSW 확정 아님)",
    "track_removed": "추적기가 트랙을 제거함",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


class TracedTracker:
    """기존 확장 process를 유지하며 실제 update의 진단 인자만 전달합니다."""
    def __init__(self, tracker):
        self.tracker = tracker
        self.trace = {}

    def __getattr__(self, name):
        return getattr(self.tracker, name)

    def update(self, *args, **kwargs):
        return self.tracker.update(*args, **kwargs, diagnostics=self.trace)


class DiagnosticRecorder:
    def __init__(self, output, metadata):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        names = ("diagnostic_frames.jsonl", "diagnostic_events.csv", "diagnostic_summary.json", "diagnostic_report.html")
        if any((self.output/name).exists() for name in names):
            raise FileExistsError("기존 진단 기록은 덮어쓰지 않습니다.")
        self.metadata = dict(metadata)
        self.frames = self.detections = self.visible_tracks = 0
        self.counts, self.stages = Counter(), Counter()
        self.samples = defaultdict(list)
        self.recent, self.previous_ids = {}, set()
        self.previous_states = {}
        self.closed = False
        self.rules = dict(overlap_iou=0.5, replacement_iou=0.5, replacement_window_frames=30,
                          html_samples_per_event=20, detector_stage="post_nms")
        self.stream = (self.output/names[0]).open("x", encoding="utf-8")
        try:
            self.event_stream = (self.output/names[1]).open("x", encoding="utf-8-sig", newline="")
            self.csv = csv.DictWriter(self.event_stream, fieldnames=["frame", "time_ms", "event", "description", "details"])
            self.csv.writeheader()
            self._write_json(self.report("running"))
        except BaseException:
            self.stream.close()
            if hasattr(self, "event_stream"):
                self.event_stream.close()
            raise

    def report(self, status):
        return dict(schema_version=1, status=status, processed_frames=self.frames,
                    total_detection_boxes=self.detections, total_visible_track_rows=self.visible_tracks,
                    event_counts=dict(self.counts), association_stages=dict(self.stages),
                    rules=self.rules, metadata=self.metadata,
                    examples=dict(self.samples), tracking_accuracy="미측정",
                    limitations=["검출은 NMS 이후이며 NMS 전 박스·실제 동물 수는 알 수 없습니다.",
                                 "겹침·ID 교체는 검토 후보입니다. 정답 없이 오검출·IDSW를 확정하지 않습니다.",
                                 "Lost 상태의 예측 박스를 실제 검출이나 출력 트랙으로 집계하지 않습니다.",
                                 "시간은 입력 FPS 기준이며 실제 촬영 FPS는 검증하지 않습니다."])

    def _write_json(self, report):
        path = self.output/"diagnostic_summary.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def event(self, frame, kind, details):
        self.counts[kind] += 1
        record = dict(frame=frame, time_ms=self.time_ms(frame), event=kind, description=EVENT_LABELS[kind], details=details)
        self.csv.writerow(dict(record, details=json.dumps(details, ensure_ascii=False, allow_nan=False)))
        if len(self.samples[kind]) < self.rules["html_samples_per_event"]:
            self.samples[kind].append(record)

    def time_ms(self, frame):
        fps = self.metadata.get("input_fps")
        return (frame-1)*1000/fps if type(fps) in (int, float) and math.isfinite(fps) and fps > 0 else None

    def add(self, frame, trace, rows):
        if self.closed or type(frame) is not int or frame != self.frames+1 or trace["tracker_frame"] != frame:
            raise ValueError("진단은 새 추적기의 1번 프레임부터 순서대로 기록해야 합니다.")
        states = {row["track_id"]: row for row in trace["states"]}
        outputs = [dict(track_id=int(row.track_id), cls=states[row.track_id]["cls"], cls_name=row.cls_name, box=list(map(float, row.box)),
                        score=float(row.score), status=row.status) for row in rows]
        payload = dict(trace, schema_version=1, frame=frame, time_ms=self.time_ms(frame), outputs=outputs)
        self.stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False)+"\n")
        self.frames += 1
        self.detections += len(trace["detections"])
        self.visible_tracks += len(outputs)
        self.stages.update(a["stage"] for a in trace["associations"])
        for collection, kind in ((trace["detections"], "detection_overlap"), (outputs, "track_overlap")):
            for a, b in combinations(collection, 2):
                if a["cls"] == b["cls"]:
                    iou = float(overlap(a["box"], [b["box"]])[0])
                    if iou >= self.rules["overlap_iou"]:
                        key = "index" if kind == "detection_overlap" else "track_id"
                        self.event(frame, kind, dict(ids=[a[key], b[key]], iou=iou))
        current = {row["track_id"]: row for row in outputs}
        if not trace["detections"]:
            self.event(frame, "empty_detections", {})
        elif not current:
            self.event(frame, "detections_without_tracks", dict(unassigned=trace["unassigned_indices"]))
        for track_id in sorted(self.previous_ids-set(current)):
            self.event(frame, "output_gap_started", dict(track_id=track_id))
        for track_id, state in states.items():
            if state["state"] == "removed":
                self.event(frame, "track_removed", dict(track_id=track_id))
        for association in trace["associations"]:
            track_id = association["track_id"]
            if (self.previous_states.get(track_id) == "lost" or association["stage"] == "reenter") and association["visible"]:
                self.event(frame, "tracker_reappearance", dict(track_id=track_id, stage=association["stage"]))
            if association["stage"] != "new":
                continue
            detection = trace["detections"][association["detection_index"]]
            for old_id, old in sorted(self.recent.items()):
                if old_id != track_id and old_id not in current and frame-old["frame"] <= self.rules["replacement_window_frames"]:
                    if old["cls"] != detection["cls"]:
                        continue
                    iou = float(overlap(old["box"], [detection["box"]])[0])
                    if iou >= self.rules["replacement_iou"]:
                        self.event(frame, "id_replacement_candidate", dict(old_id=old_id, new_id=track_id,
                                   gap_frames=frame-old["frame"], iou=iou, new_visible=association["visible"]))
        for track_id, row in current.items():
            self.recent[track_id] = dict(row, frame=frame)
        self.recent = {key: value for key, value in self.recent.items()
                       if frame-value["frame"] <= self.rules["replacement_window_frames"]}
        self.previous_ids = set(current)
        self.previous_states = {key: value["state"] for key, value in states.items()}
        self.stream.flush()
        self.event_stream.flush()
        if frame % 25 == 0:
            self._write_json(self.report("running"))

    def finish(self, status, metadata=None):
        if self.closed:
            return
        if status not in ("completed", "failed"):
            raise ValueError("최종 상태는 completed 또는 failed이어야 합니다.")
        self.stream.close()
        self.event_stream.close()
        if metadata:
            self.metadata.update(metadata)
        report = self.report(status)
        counts = "".join(f"<tr><td>{html.escape(EVENT_LABELS[kind])}</td><td>{count}</td></tr>"
                         for kind, count in sorted(self.counts.items()))
        if not counts:
            counts = "<tr><td colspan='2'>설정한 규칙의 검토 후보 없음. 정확도를 보증하지 않습니다.</td></tr>"
        examples = sorted((row for rows in self.samples.values() for row in rows), key=lambda row: row["frame"])
        lines = "".join(f"<tr><td>{row['frame']}</td><td>{row['time_ms'] if row['time_ms'] is not None else '—'}</td><td>{html.escape(row['description'])}</td>"
                        f"<td>{html.escape(json.dumps(row['details'], ensure_ascii=False))}</td></tr>" for row in examples)
        page = ("<!doctype html><html lang='ko'><meta charset='utf-8'><title>보호색 추적 진단</title>"
                "<style>body{font-family:sans-serif;margin:28px;max-width:1200px}table{border-collapse:collapse}"
                "td,th{border:1px solid #aaa;padding:8px;text-align:left}</style>"
                f"<h1>보호색 추적 진단</h1><p>상태: {html.escape(status)} · 기록 프레임: {self.frames}</p>"
                "<p>의심 구간의 검토 자료입니다. 정답 기반 정확도·ID Switches는 측정하지 않았습니다.</p>"
                "<p>diagnostic.mp4의 왼쪽은 NMS 후 검출, 오른쪽은 출력 트랙입니다. GUI의 진단 영상 재생을 사용하세요.</p>"
                "<p><a href='diagnostic_frames.jsonl'>전체 프레임 기록</a> · <a href='diagnostic_events.csv'>전체 사건 CSV</a> · "
                "<a href='diagnostic_summary.json'>설정·요약 JSON</a></p>"
                f"<table><tr><th>검토 유형</th><th>관측 건수 (반복 포함)</th></tr>{counts}</table>"
                "<details><summary>실행 설정·파일 해시</summary><pre>"+
                html.escape(json.dumps(self.metadata, ensure_ascii=False, indent=2))+"</pre></details>"
                f"<h2>유형별 최대 20건 예시</h2><table><tr><th>프레임</th><th>입력 시간(ms)</th><th>유형</th><th>상세</th></tr>{lines}</table>"
                "<h2>해석의 한계</h2><ul>"+"".join(f"<li>{html.escape(note)}</li>" for note in report["limitations"])+"</ul></html>")
        target = self.output/"diagnostic_report.html"
        temporary = target.with_suffix(".html.tmp")
        temporary.write_text(page, encoding="utf-8")
        temporary.replace(target)
        self._write_json(report)
        self.closed = True


def draw_detections(frame, trace):
    import cv2
    panel = frame.copy()
    font_scale = max(0.5, panel.shape[1]/1280)
    associations = {row["detection_index"]: row for row in trace["associations"]}
    for detection in trace["detections"]:
        x1, y1, x2, y2 = map(int, detection["box"])
        text = f"D{detection['index']} {detection['score']:.2f} {detection['group']}"
        linked = associations.get(detection["index"])
        if linked:
            text += f" -> ID{linked['track_id']} ({linked['stage']})"
            if not linked["visible"]:
                text += " pending"
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
        text_x = max(0, min(x1, panel.shape[1]-text_width))
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(panel, text, (text_x, max(40, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 220, 255), 2)
    return label_panel(panel, f"FRAME {trace['tracker_frame']} | POST-NMS DETECTIONS")


def label_panel(panel, text):
    import cv2
    scale = max(0.5, panel.shape[1]/1280)
    height = max(28, round(35*scale))
    cv2.rectangle(panel, (0, 0), (panel.shape[1], height), (0, 0, 0), -1)
    cv2.putText(panel, text, (6, height-8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return panel
