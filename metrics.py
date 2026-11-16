"""MOT 추적 성능 지표: HOTA, MOTA, IDF1, ID Switches (요구 사항 7).

TrackEval(MotChallenge2DBox)의 계산 방식을 그대로 옮겼습니다. 외부 저장소 없이 평가·튜닝·GUI 비교를
돌리기 위한 것이며, 공식 수치가 필요하면 ``evaluate_mot.py``(TrackEval)로 교차 확인할 수 있습니다.

* 정답: ``flag == 1``이고 ``class == 1``(보행자)인 박스만 평가합니다.
* 전처리: 정지한 사람·탈것 위 사람·반사 등 distractor 정답(class 2, 7, 8, 12)과 IoU 0.5 이상으로
  연결된 추적 결과는 FP로 세지 않도록 제거합니다.
* CLEAR(MOTA, IDSW): IoU 0.5 이상, 직전 프레임 연결을 우선하는 헝가리안 할당.
* Identity(IDF1): 전체 영상에서 정답 ID ↔ 추적 ID를 1:1로 묶었을 때 맞은 검출 수.
* HOTA: IoU 임계값 0.05~0.95(19개)마다 검출 정확도(DetA)와 연관 정확도(AssA)의 기하평균을 내고 평균합니다.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.matching import iou_matrix

DISTRACTOR_CLASSES = {2, 7, 8, 12}
ALPHAS = np.arange(0.05, 0.99, 0.05)
EPS = np.finfo(float).eps


@dataclass
class Frame:
    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    visibility: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _read_rows(path):
    rows = []
    with open(path, encoding="utf-8-sig") as source:
        for line in source:
            if line.strip():
                rows.append([float(v) for v in line.replace(" ", "").split(",")])
    return rows


def load_ground_truth(path, frames=None):
    """정답 파일을 ``(평가 대상 {frame: Frame}, distractor {frame: boxes})``로 읽습니다."""
    valid, distractors = defaultdict(list), defaultdict(list)
    for row in _read_rows(path):
        frame = int(row[0])
        if frames is not None and frame not in frames:
            continue
        box = [row[2], row[3], row[2] + row[4], row[3] + row[5]]
        flag = row[6] if len(row) > 6 else 1
        cls = int(row[7]) if len(row) > 7 else 1
        visibility = row[8] if len(row) > 8 else 1.0
        if cls in DISTRACTOR_CLASSES:
            distractors[frame].append(box)
        elif flag == 1 and cls == 1:
            valid[frame].append((int(row[1]), box, visibility))
    gt = {frame: Frame(np.array([r[0] for r in items]), np.array([r[1] for r in items]),
                       np.array([r[2] for r in items])) for frame, items in valid.items()}
    return gt, {frame: np.array(boxes) for frame, boxes in distractors.items()}


def load_results(path, frames=None):
    items = defaultdict(list)
    for row in _read_rows(path):
        frame = int(row[0])
        if frames is None or frame in frames:
            items[frame].append((int(row[1]), [row[2], row[3], row[2] + row[4], row[3] + row[5]]))
    return {frame: Frame(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))
            for frame, rows in items.items()}


def remove_distractor_matches(gt, distractors, results):
    """distractor 정답과 연결된 추적 박스를 제거합니다 (TrackEval 전처리와 같음)."""
    cleaned = {}
    for frame, tracked in results.items():
        extra = distractors.get(frame)
        if extra is None or not len(extra) or not len(tracked.ids):
            cleaned[frame] = tracked
            continue
        valid = gt.get(frame, Frame())
        all_gt = np.vstack([valid.boxes.reshape(-1, 4), extra])
        iou = iou_matrix(all_gt, tracked.boxes)
        score = np.where(iou >= 0.5 - EPS, iou, 0)
        rows, columns = linear_sum_assignment(-score)
        drop = {c for r, c in zip(rows, columns) if score[r, c] > 0 and r >= len(valid.ids)}
        keep = np.array([i not in drop for i in range(len(tracked.ids))], dtype=bool)
        cleaned[frame] = Frame(tracked.ids[keep], tracked.boxes[keep])
    return cleaned


def evaluate_sequence(gt, results, frame_numbers, events=None):
    """한 시퀀스의 원시 카운트를 계산합니다. ``events`` 리스트를 주면 ID 변경·끊김 사례를 기록합니다."""
    gt_ids = sorted({i for f in gt.values() for i in f.ids.tolist()})
    tr_ids = sorted({i for f in results.values() for i in f.ids.tolist()})
    gt_index = {v: i for i, v in enumerate(gt_ids)}
    tr_index = {v: i for i, v in enumerate(tr_ids)}
    num_gt, num_tr = len(gt_ids), len(tr_ids)
    empty = Frame()

    # ---- HOTA 1단계: 전체 영상 기준 정답-추적 ID 쌍의 정렬 점수
    potential = np.zeros((num_gt, num_tr))
    gt_count, tr_count = np.zeros(num_gt), np.zeros(num_tr)
    similarity = {}
    for frame in frame_numbers:
        g, t = gt.get(frame, empty), results.get(frame, empty)
        gi = np.array([gt_index[v] for v in g.ids.tolist()], dtype=int)
        ti = np.array([tr_index[v] for v in t.ids.tolist()], dtype=int)
        sim = iou_matrix(g.boxes, t.boxes)
        similarity[frame] = (gi, ti, sim)
        if len(gi) and len(ti):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.where(denom > EPS, sim / np.maximum(denom, EPS), 0)
            potential[gi[:, None], ti[None, :]] += sim_iou
        gt_count[gi] += 1
        tr_count[ti] += 1
    global_score = potential / np.maximum(gt_count[:, None] + tr_count[None, :] - potential, EPS)

    hota_tp = np.zeros(len(ALPHAS))
    hota_fn = np.zeros(len(ALPHAS))
    hota_fp = np.zeros(len(ALPHAS))
    loc_sum = np.zeros(len(ALPHAS))
    match_counts = np.zeros((len(ALPHAS), num_gt, num_tr))

    # ---- CLEAR / Identity 누적값
    clear = dict(tp=0, fn=0, fp=0, idsw=0, frag=0, gt_dets=0, tr_dets=0)
    identity = np.zeros((num_gt, num_tr))
    previous = np.full(num_gt, -1)         # 각 정답이 마지막으로 연결된 추적 ID
    previous_frame = np.full(num_gt, -1)   # 직전 프레임 연결 (우선 연결용)
    tracked_before = np.zeros(num_gt, dtype=bool)
    was_tracked = np.zeros(num_gt, dtype=bool)

    for step, frame in enumerate(frame_numbers):
        gi, ti, sim = similarity[frame]
        clear["gt_dets"] += len(gi)
        clear["tr_dets"] += len(ti)
        if not len(gi) or not len(ti):
            hota_fn += len(gi)
            hota_fp += len(ti)
            clear["fn"] += len(gi)
            clear["fp"] += len(ti)
            was_tracked[:] = False
            continue

        # HOTA 2단계: 정렬 점수 × IoU로 프레임별 할당
        rows, columns = linear_sum_assignment(-(global_score[gi][:, ti] * sim))
        for a, alpha in enumerate(ALPHAS):
            ok = sim[rows, columns] >= alpha - EPS
            r, c = rows[ok], columns[ok]
            hota_tp[a] += len(r)
            hota_fn[a] += len(gi) - len(r)
            hota_fp[a] += len(ti) - len(r)
            loc_sum[a] += sim[r, c].sum()
            match_counts[a, gi[r], ti[c]] += 1

        # CLEAR: 직전 프레임과 같은 짝을 우선하는 할당
        score = np.where(sim >= 0.5 - EPS, 1000 * (previous_frame[gi][:, None] == ti[None, :]) + sim, 0)
        rows, columns = linear_sum_assignment(-score)
        ok = score[rows, columns] > 0
        rows, columns = rows[ok], columns[ok]
        matched_gt, matched_tr = gi[rows], ti[columns]
        switched = (previous[matched_gt] >= 0) & (previous[matched_gt] != matched_tr)
        fragmented = tracked_before[matched_gt] & ~was_tracked[matched_gt]
        if events is not None:
            for g, t, s, f in zip(matched_gt, matched_tr, switched, fragmented):
                if s:
                    events.append(dict(type="IDSW", frame=frame, gt_id=gt_ids[g],
                                       old_id=tr_ids[previous[g]], new_id=tr_ids[t]))
                elif f:
                    events.append(dict(type="FRAG", frame=frame, gt_id=gt_ids[g],
                                       old_id=tr_ids[previous[g]], new_id=tr_ids[t]))
        clear["idsw"] += int(switched.sum())
        clear["frag"] += int(fragmented.sum())
        clear["tp"] += len(rows)
        clear["fn"] += len(gi) - len(rows)
        clear["fp"] += len(ti) - len(rows)
        was_tracked[:] = False
        was_tracked[matched_gt] = True
        tracked_before[matched_gt] = True
        previous[matched_gt] = matched_tr
        previous_frame[:] = -1
        previous_frame[matched_gt] = matched_tr

        # Identity: IoU 0.5 이상인 모든 (정답, 추적) 쌍 수를 셉니다.
        g_hit, t_hit = np.nonzero(sim >= 0.5 - EPS)
        np.add.at(identity, (gi[g_hit], ti[t_hit]), 1)

    # HOTA 연관 정확도
    ass_a = np.zeros(len(ALPHAS))
    for a in range(len(ALPHAS)):
        counts = match_counts[a]
        ass = counts / np.maximum(gt_count[:, None] + tr_count[None, :] - counts, 1)
        ass_a[a] = (counts * ass).sum() / max(1.0, hota_tp[a])

    idtp = 0.0
    if num_gt and num_tr:
        rows, columns = linear_sum_assignment(-identity)
        idtp = float(identity[rows, columns].sum())

    return dict(hota_tp=hota_tp, hota_fn=hota_fn, hota_fp=hota_fp, loc_sum=loc_sum,
                ass_a=ass_a, idtp=idtp, num_gt_ids=num_gt, num_tr_ids=num_tr, **clear)


def combine(raw_list):
    """여러 시퀀스의 원시 카운트를 TrackEval과 같은 방식으로 합쳐 최종 지표(%)를 만듭니다."""
    tp = sum(r["hota_tp"] for r in raw_list)
    fn = sum(r["hota_fn"] for r in raw_list)
    fp = sum(r["hota_fp"] for r in raw_list)
    ass = sum(r["ass_a"] * r["hota_tp"] for r in raw_list) / np.maximum(tp, 1)
    det = tp / np.maximum(tp + fn + fp, 1)
    loc = sum(r["loc_sum"] for r in raw_list) / np.maximum(tp, 1)
    hota = np.sqrt(det * ass)
    total = {k: sum(r[k] for r in raw_list) for k in ("tp", "fn", "fp", "idsw", "frag",
                                                       "gt_dets", "tr_dets", "idtp")}
    mota = 1 - (total["fn"] + total["fp"] + total["idsw"]) / max(1, total["gt_dets"])
    idf1 = 2 * total["idtp"] / max(1, total["gt_dets"] + total["tr_dets"])
    return {
        "HOTA": 100 * float(hota.mean()), "DetA": 100 * float(det.mean()),
        "AssA": 100 * float(ass.mean()), "LocA": 100 * float(loc.mean()),
        "MOTA": 100 * mota, "IDF1": 100 * idf1, "IDSW": int(total["idsw"]),
        "Frag": int(total["frag"]), "FP": int(total["fp"]), "FN": int(total["fn"]),
        "GT_Dets": int(total["gt_dets"]), "IDs": int(sum(r["num_tr_ids"] for r in raw_list)),
        "GT_IDs": int(sum(r["num_gt_ids"] for r in raw_list)),
    }


def evaluate_files(gt_path, result_path, frame_numbers, events=None):
    gt, distractors = load_ground_truth(gt_path, set(frame_numbers))
    results = remove_distractor_matches(gt, distractors, load_results(result_path, set(frame_numbers)))
    return evaluate_sequence(gt, results, frame_numbers, events)
