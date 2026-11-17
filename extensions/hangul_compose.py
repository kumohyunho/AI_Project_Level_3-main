"""떠다니던 자모가 모여 글자 모양을 이루면 한 글자로 조합하고, 글자를 이어 단어로 읽습니다.  설계: ``docs/HANGUL.md``

추적 결과(트랙 ID, 다수결로 정한 자모, 박스)만 보고 판단하는 위치 규칙입니다. 학습하는 부분은 없습니다.

1. **중성(모음)을 기준**으로 글자를 만듭니다. 모음마다 모양에 맞는 자리에 있는 자음을 초성으로 붙입니다.
   * 세로 모음(ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅣ): 초성은 모음 **왼쪽**
   * 가로 모음(ㅗ ㅛ ㅜ ㅠ ㅡ): 초성은 모음 **위쪽**
   * 섞인 모음(ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ ㅢ): 초성은 모음 **왼쪽 위**
2. 초성과 중성을 감싼 영역 **바로 아래**에 있는 자음을 종성으로 붙입니다 (ㄸ ㅃ ㅉ은 종성이 될 수 없음).
3. 같은 줄에서 왼쪽부터 글자를 잇고, 글자 사이가 많이 벌어지면 단어를 나눕니다.
4. 같은 자리에서 최근 30프레임 중 가장 많이 읽힌 단어를, ``min_frames``번 이상 읽혔을 때만 표시합니다.
   떠다니던 자모가 우연히 잠깐 맞물린 것은 버리고, 자모 하나가 잠깐 다르게 읽혀도 표시가 깜빡이지 않습니다.

거리 기준은 모두 자모 크기(박스 긴 변의 중앙값)에 대한 비율이라 화면 해상도와 글자 크기에 무관합니다.
자모가 크게 기울어져 있으면(약 ±20° 넘게) 위·아래·왼쪽 판단이 틀릴 수 있습니다.
"""
from dataclasses import dataclass, field

import numpy as np

CHOSEONG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSEONG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSEONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
             "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
VERTICAL = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅣ")
HORIZONTAL = set("ㅗㅛㅜㅠㅡ")
MIXED = set("ㅘㅙㅚㅝㅞㅟㅢ")


def compose(cho, jung, jong=""):
    """초성·중성(·종성) 자모를 한 글자로 합칩니다. 예: compose("ㅎ", "ㅏ", "ㄴ") → "한"."""
    if cho not in CHOSEONG or jung not in JUNGSEONG or jong not in JONGSEONG:
        raise ValueError(f"조합할 수 없는 자모: {cho!r} {jung!r} {jong!r}")
    return chr(0xAC00 + (CHOSEONG.index(cho) * 21 + JUNGSEONG.index(jung)) * 28 + JONGSEONG.index(jong))


def decompose(syllable):
    """한 글자를 (초성, 중성, 종성) 자모로 나눕니다. 예: decompose("한") → ("ㅎ", "ㅏ", "ㄴ")."""
    code = ord(syllable) - 0xAC00
    if not 0 <= code < 11172:
        raise ValueError(f"완성형 한글 글자가 아닙니다: {syllable!r}")
    return CHOSEONG[code // 588], JUNGSEONG[(code % 588) // 28], JONGSEONG[code % 28]


@dataclass
class Jamo:
    id: int
    char: str
    box: tuple   # x1, y1, x2, y2

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass
class Syllable:
    text: str
    box: tuple
    members: tuple   # 자모 트랙 ID (초성, 중성, 종성 순)


@dataclass
class Word:
    text: str
    box: tuple
    members: tuple
    syllables: list = field(default_factory=list)


def _union(boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return (float(boxes[:, 0].min()), float(boxes[:, 1].min()), float(boxes[:, 2].max()), float(boxes[:, 3].max()))


def _box_gap(a, b):
    """두 박스 사이의 빈틈 거리 (겹치면 0)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def _initial_cost(consonant, vowel, size, reach):
    """``consonant``가 ``vowel``의 초성 자리에 있으면 거리(작을수록 좋음), 아니면 ``None``."""
    cx, cy = consonant.center
    vx, vy = vowel.center
    x1, y1, x2, y2 = vowel.box
    gap = _box_gap(consonant.box, vowel.box)
    if gap > reach * size:
        return None
    if vowel.char in VERTICAL:
        # 왼쪽에 있고, 높이는 모음의 위쪽~가운데
        if not (cx < x1 + 0.2 * (x2 - x1) and y1 - 0.3 * size <= cy <= vy + 0.25 * size):
            return None
    elif vowel.char in HORIZONTAL:
        # 위쪽에 있고, 가로 위치는 모음 막대 범위 안
        if not (cy < y1 + 0.2 * (y2 - y1) and x1 - 0.3 * size <= cx <= x2 + 0.3 * size):
            return None
    else:
        # 섞인 모음: 왼쪽 위 (모음 박스의 가로 가운데보다 왼쪽, 세로 가운데보다 위)
        if not (cx < vx and cy < vy):
            return None
    return gap


def _final_cost(consonant, upper, size, reach):
    """``consonant``가 ``upper``(초성+중성 영역) 바로 아래 종성 자리에 있으면 거리, 아니면 ``None``."""
    cx, cy = consonant.center
    x1, y1, x2, y2 = upper
    if consonant.char not in JONGSEONG or consonant.char == "":
        return None
    if not (cy > y2 - 0.25 * size and x1 - 0.3 * size <= cx <= x2 + 0.3 * size):
        return None
    gap = max(0.0, consonant.box[1] - y2)
    if gap > reach * size:
        return None
    return gap


def group_syllables(items, reach=0.8):
    """자모 목록을 글자로 묶습니다. 반환: (글자 목록, 글자에 쓰이지 않은 자모 목록).

    ``reach``: 초성·종성이 모음(또는 초성+중성 영역)에서 떨어져 있어도 되는 최대 빈틈 (자모 크기 대비).
    """
    items = [j for j in items if j.char]
    if not items:
        return [], []
    size = float(np.median([max(j.box[2] - j.box[0], j.box[3] - j.box[1]) for j in items]))
    vowels = [j for j in items if j.char in JUNGSEONG]
    consonants = [j for j in items if j.char in CHOSEONG or j.char in JONGSEONG]

    # 1) 초성: 모든 (모음, 자음) 후보 쌍을 거리순으로 보고, 아직 쓰이지 않은 것끼리 짝짓습니다.
    pairs = []
    for vi, vowel in enumerate(vowels):
        for ci, consonant in enumerate(consonants):
            if consonant.char not in CHOSEONG:
                continue
            cost = _initial_cost(consonant, vowel, size, reach)
            if cost is not None:
                pairs.append((cost, vi, ci))
    initial, used = {}, set()
    for cost, vi, ci in sorted(pairs):
        if vi in initial or ci in used:
            continue
        initial[vi] = ci
        used.add(ci)

    # 2) 종성: 초성이 붙은 글자마다 바로 아래의 남은 자음
    finals, pairs = {}, []
    for vi, ci in initial.items():
        upper = _union([vowels[vi].box, consonants[ci].box])
        for fi, consonant in enumerate(consonants):
            if fi in used:
                continue
            cost = _final_cost(consonant, upper, size, reach)
            if cost is not None:
                pairs.append((cost, vi, fi))
    for cost, vi, fi in sorted(pairs):
        if vi in finals or fi in used:
            continue
        finals[vi] = fi
        used.add(fi)

    syllables = []
    for vi, ci in initial.items():
        members = [consonants[ci], vowels[vi]] + ([consonants[finals[vi]]] if vi in finals else [])
        text = compose(members[0].char, members[1].char, members[2].char if len(members) == 3 else "")
        syllables.append(Syllable(text, _union([m.box for m in members]), tuple(m.id for m in members)))
    member_ids = {i for s in syllables for i in s.members}
    leftover = [j for j in items if j.id not in member_ids]
    return syllables, leftover


def group_words(syllables, space=0.6):
    """글자를 줄별로 왼쪽부터 이어 단어로 만듭니다.

    ``space``: 이웃 글자 사이 빈틈이 글자 폭의 이 비율보다 크면 다른 단어로 나눕니다.
    """
    if not syllables:
        return []
    heights = [s.box[3] - s.box[1] for s in syllables]
    widths = [s.box[2] - s.box[0] for s in syllables]
    line_tol, width = 0.5 * float(np.median(heights)), float(np.median(widths))
    lines = []
    for syllable in sorted(syllables, key=lambda s: (s.box[1] + s.box[3]) / 2):
        cy = (syllable.box[1] + syllable.box[3]) / 2
        for line in lines:
            if abs(line["cy"] - cy) <= line_tol:
                line["items"].append(syllable)
                line["cy"] = float(np.mean([(s.box[1] + s.box[3]) / 2 for s in line["items"]]))
                break
        else:
            lines.append({"cy": cy, "items": [syllable]})
    words = []
    for line in sorted(lines, key=lambda l: l["cy"]):
        current = []
        for syllable in sorted(line["items"], key=lambda s: s.box[0]):
            if current and syllable.box[0] - current[-1].box[2] > space * width:
                words.append(current)
                current = []
            current.append(syllable)
        words.append(current)
    return [Word("".join(s.text for s in w), _union([s.box for s in w]),
                 tuple(i for s in w for i in s.members), list(w)) for w in words]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class WordStabilizer:
    """프레임마다 읽은 단어를 자리별로 모아, 최근 ``window``프레임 중 가장 많이 읽힌 단어를 표시합니다.

    그 단어가 ``min_frames``번 이상 읽혀야 표시하므로, 떠다니던 자모가 우연히 잠깐 맞물린 것은 버려집니다.
    자모 하나가 몇 프레임 다르게 읽혀도(예: ㅏ가 ㅑ로) 표시가 깜빡이지 않습니다.
    표시된 자리에는 ``first_id``부터 번호를 붙이고, 그 자리에 단어가 이어지는 동안 번호를 유지합니다.
    ``max_missing``프레임 넘게 읽히지 않으면 지웁니다.
    """

    def __init__(self, min_frames=8, window=30, max_missing=5, first_id=1001, match_iou=0.3):
        self.min_frames, self.window, self.max_missing, self.match_iou = min_frames, window, max_missing, match_iou
        self.next_id = first_id
        self.entries = []   # dict(box, history, missing, id, words)

    def update(self, words):
        from collections import Counter, deque

        matched = set()
        for word in words:
            best, best_iou = None, self.match_iou
            for index, entry in enumerate(self.entries):
                overlap = _iou(entry["box"], word.box)
                if index not in matched and overlap >= best_iou:
                    best, best_iou = index, overlap
            if best is None:
                self.entries.append(dict(box=word.box, history=deque(maxlen=self.window), missing=0, id=None, words={}))
                best = len(self.entries) - 1
            entry = self.entries[best]
            matched.add(best)
            entry["history"].append(word.text)
            entry["words"][word.text] = word
            entry.update(box=word.box, missing=0)
        for index, entry in enumerate(self.entries):
            if index not in matched:
                entry["missing"] += 1
                entry["history"].append(None)
        self.entries = [e for e in self.entries if e["missing"] <= self.max_missing]
        shown = []
        for entry in self.entries:
            counts = Counter(t for t in entry["history"] if t is not None)
            if not counts:
                continue
            text, count = counts.most_common(1)[0]
            if count < self.min_frames:
                continue
            if entry["id"] is None:
                entry["id"] = self.next_id
                self.next_id += 1
            word = entry["words"][text]
            shown.append((entry["id"], Word(text, entry["box"], word.members, word.syllables)))
        return sorted(shown, key=lambda item: item[0])


def read_words(items, reach=0.8, space=0.6):
    """자모 목록 → 단어 목록 (안정화 없이 이 프레임만 보고 읽음)."""
    syllables, _ = group_syllables(items, reach)
    return group_words(syllables, space)
