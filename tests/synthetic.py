"""정답을 알고 있는 합성 장면: 교차·가림·화면 이탈 후 재진입이 모두 들어 있습니다.

실제 데이터 없이도 추적기·지표 코드를 검증하고, 기본판과 개선판의 차이를 재현 가능하게 보여 주기 위한 것입니다.
사람은 서로 다른 옷 색(상의/하의)을 가진 직사각형으로 그리고, 검출은 정답 박스에 잡음을 섞어 만듭니다.
가림 구간에서는 검출을 지우거나 점수를 낮춥니다.
"""
import numpy as np

WIDTH, HEIGHT = 640, 360
PILLAR = (300, 0, 360, 360)   # 화면 가운데 기둥: 지나가는 사람을 가립니다.


def _people(frames):
    """(id, 상의 BGR, 하의 BGR, 시작 x, 끝 x, y, 등장 프레임, 퇴장 프레임) 목록."""
    return [
        (1, (40, 40, 220), (60, 60, 60), 40, 600, 150, 1, frames),         # 왼→오, 기둥 뒤 통과
        (2, (220, 120, 30), (30, 160, 30), 600, 40, 160, 1, frames),       # 오→왼, 1번과 교차
        (3, (30, 200, 230), (120, 30, 120), 120, 220, 60, 1, frames),      # 천천히 걷는 사람
        (4, (200, 60, 200), (40, 120, 200), 500, 700, 250, 1, frames // 2),  # 오른쪽 밖으로 나감
        (4, (200, 60, 200), (40, 120, 200), 700, 480, 250, frames // 2 + 25, frames),  # 다시 들어옴
    ]


def make_sequence(frames=120, seed=0):
    """(이미지 목록, 정답 {frame: [(id, box, visibility)]}, 검출 {frame: (boxes, scores)})."""
    rng = np.random.default_rng(seed)
    images, gt, detections = [], {}, {}
    people = _people(frames)
    for number in range(1, frames + 1):
        image = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
        image[:, :, 1] = 100
        objects = []
        for pid, top, bottom, x0, x1, y, start, end in people:
            if not start <= number <= end:
                continue
            t = (number - start) / max(1, end - start)
            cx = x0 + (x1 - x0) * t
            box = np.array([cx - 15, y, cx + 15, y + 80], dtype=float)
            if box[2] < 0 or box[0] > WIDTH:
                continue
            objects.append((pid, box, top, bottom))
        # 뒤에서부터 그려 y가 큰(가까운) 사람이 앞을 가리게 합니다.
        for pid, box, top, bottom in sorted(objects, key=lambda o: o[1][1]):
            x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(int)
            image[y1:(y1 + y2) // 2, x1:x2] = top
            image[(y1 + y2) // 2:y2, x1:x2] = bottom
        image[:, PILLAR[0]:PILLAR[2]] = (70, 70, 70)
        frame_gt, boxes, scores = [], [], []
        for pid, box, _, _ in objects:
            clipped = np.clip(box, [0, 0, 0, 0], [WIDTH, HEIGHT, WIDTH, HEIGHT])
            area = (box[2] - box[0]) * (box[3] - box[1])
            visible = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1]) / area
            overlap = max(0.0, min(clipped[2], PILLAR[2]) - max(clipped[0], PILLAR[0]))
            visible *= 1 - overlap / max(1.0, clipped[2] - clipped[0])
            frame_gt.append((pid, box, visible))
            if visible < 0.3:
                continue  # 거의 가려져 검출되지 않음
            noisy = clipped + rng.normal(0, 1.5, 4)
            boxes.append(noisy)
            scores.append(0.9 if visible > 0.7 else 0.35)  # 일부 가려지면 점수가 떨어짐
        gt[number] = frame_gt
        detections[number] = (np.array(boxes).reshape(-1, 4), np.array(scores))
        images.append(image)
    return images, gt, detections


def write_gt(path, gt):
    """MOT 정답 형식으로 저장합니다. MOT 좌표는 1부터 시작하므로 +1 합니다 (results_io.write_mot과 같음)."""
    with open(path, "w", encoding="utf-8") as target:
        for frame, items in gt.items():
            for pid, box, visible in items:
                x1, y1, x2, y2 = box
                target.write(f"{frame},{pid},{x1 + 1:.2f},{y1 + 1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,1,{visible:.3f}\n")
