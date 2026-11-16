"""객체별 위치 변화 기록과 이동 방향·거리·속도·궤적 계산 (요구 사항 5)."""
from collections import deque
from dataclasses import dataclass, field
from math import atan2, degrees, hypot

ARROWS = ("→", "↘", "↓", "↙", "←", "↖", "↑", "↗")


@dataclass
class MotionState:
    trail: deque = field(default_factory=lambda: deque(maxlen=45))
    distance: float = 0.0
    direction: str = "—"
    speed: float = 0.0          # px/s, 최근 이동의 지수 평균
    first_frame: int = 0
    last_frame: int = 0


class MotionRecorder:
    """박스 하단 중앙(발 위치)이 아닌 박스 중심을 기준으로 누적 이동을 계산합니다."""

    def __init__(self, fps=30.0, trail_length=45):
        self.fps = fps if fps and fps > 0 else 30.0
        self.trail_length = trail_length
        self.states = {}

    def reset(self):
        self.states.clear()

    def update(self, track_id, box, frame_index):
        x1, y1, x2, y2 = box
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        state = self.states.get(track_id)
        if state is None:
            state = MotionState(trail=deque(maxlen=self.trail_length), first_frame=frame_index)
            self.states[track_id] = state
        if state.trail:
            last_x, last_y = state.trail[-1]
            dx, dy = x - last_x, y - last_y
            step = hypot(dx, dy)
            gap = max(1, frame_index - state.last_frame)
            state.distance += step
            state.speed = 0.7 * state.speed + 0.3 * step * self.fps / gap
            if step >= 1:
                state.direction = ARROWS[round(degrees(atan2(dy, dx)) / 45) % 8]
        state.trail.append((x, y))
        state.last_frame = frame_index
        return state
