"""확장 기능(한글 자모) 시연 영상: 떠다니던 자모가 모여 단어가 되었다가 다시 흩어집니다.  설계: ``docs/HANGUL.md``

GUI에서 ``hangul_demo.mp4``를 열고 **확장 → 한글 자모**를 켜면, 자모마다 ID·자모가 붙어 추적되다가
자모가 모여 단어 모양이 되는 동안 단어(예: ``한글``)가 함께 표시됩니다.

진행 (기본, 30fps)
1. 떠다님: 모든 자모가 회전하며 화면을 떠다닙니다.
2. 모임: 첫 단어의 자모가 화면 가운데 글자 자리로 날아가고, 나머지는 위·아래 띠로 비켜납니다.
3. 멈춤: 단어 모양으로 잠시 멈춥니다.
4. 흩어짐: 다시 떠다닙니다. 다음 단어로 2~4를 반복합니다.

``hangul_demo.json``에 단계별 프레임 범위와 그 단계에 화면에 있어야 할 단어(정답)를 기록합니다.

예)
    python hangul_demo.py                                    # → hangul_demo.mp4, hangul_demo.json
    python hangul_demo.py --words 한글 객체추적 인공지능 --output my_demo.mp4

모든 글자는 자모 40종(쌍자음·복합 모음 포함)으로 만들 수 있어야 합니다. 겹받침(ㄳ, ㄺ 등)은 쓸 수 없습니다.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from extensions.hangul import CHAR
from extensions.hangul_compose import HORIZONTAL, VERTICAL, decompose
from hangul_dataset import _crop, find_fonts, paste, rotate, split_fonts

ROOT = Path(__file__).resolve().parent
JAMO_SET = set(CHAR.values())
DISTRACTORS = "ㅂㅅㅇㅕㅍㅛㅁㅑ"


def syllable_layout(syllable, x0, yc, g):
    """한 글자의 자모 목표 위치 [(자모, 중심 x, 중심 y)]와 글자 폭.

    ``g``: 자음 그림의 크기(픽셀). 위치는 extensions/hangul_compose.py의 조합 규칙에 맞춥니다.
    """
    cho, jung, jong = decompose(syllable)
    for char in (cho, jung, jong):
        if char and char not in JAMO_SET:
            raise ValueError(f"'{syllable}'의 자모 {char}는 검출 대상 40종에 없습니다 (겹받침 등).")
    if jung in HORIZONTAL:   # 초성은 위, 중성은 아래, 종성은 맨 아래
        if jong:
            parts = [(cho, x0 + 1.0 * g, yc - 0.95 * g), (jung, x0 + 1.0 * g, yc - 0.05 * g), (jong, x0 + 1.0 * g, yc + 0.85 * g)]
        else:
            parts = [(cho, x0 + 1.0 * g, yc - 0.5 * g), (jung, x0 + 1.0 * g, yc + 0.4 * g)]
    else:   # 세로 모음: 초성은 왼쪽 (섞인 모음은 왼쪽 위), 종성은 아래
        lift = 0.45 * g if jong else 0.0
        cho_y = yc - lift - (0.3 * g if jung not in VERTICAL else 0.0)
        parts = [(cho, x0 + 0.5 * g, cho_y), (jung, x0 + 1.45 * g, yc - lift)]
        if jong:
            parts.append((jong, x0 + 0.95 * g, yc + 0.8 * g))
    return parts, 2.0 * g


def word_layout(word, width, height, g):
    """단어를 화면 가운데에 놓았을 때 자모별 목표 위치."""
    total = sum(syllable_layout(ch, 0, 0, g)[1] for ch in word)
    x, yc, parts = (width - total) / 2, height / 2, []
    for syllable in word:
        placed, advance = syllable_layout(syllable, x, yc, g)
        parts += placed
        x += advance
    return parts


def render(char, font_path, size, color):
    """배경판 없이 자모 하나를 그립니다 (외곽선으로 어떤 배경에서도 잘 보이게)."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, size)
    left, top, right, bottom = font.getbbox(char, stroke_width=2)
    side = max(right - left, bottom - top) + 12
    layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    outline = (255, 255, 255) if sum(color) < 380 else (20, 20, 20)
    ImageDraw.Draw(layer).text(((side - (right - left)) // 2 - left, (side - (bottom - top)) // 2 - top), char,
                               font=font, fill=color + (255,), stroke_width=2, stroke_fill=outline + (255,))
    return _crop(layer, margin=2)


def background(width, height, rng):
    """부드러운 그라데이션 + 흐린 원 몇 개."""
    import cv2

    top = np.array([rng.randint(150, 230) for _ in range(3)], dtype=np.float32)
    bottom = np.array([rng.randint(60, 140) for _ in range(3)], dtype=np.float32)
    t = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    image = (top * (1 - t) + bottom * t).repeat(width, axis=1)
    for _ in range(6):
        center = (rng.randint(0, width), rng.randint(0, height))
        cv2.circle(image, center, rng.randint(60, 180), [rng.randint(80, 220) for _ in range(3)], -1)
    return cv2.GaussianBlur(np.clip(image, 0, 255).astype(np.uint8), (0, 0), 25)


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, t)))


def make_demo(words, output, width=960, height=540, fps=30, size=96, seed=3, font=None,
              float_frames=150, gather_frames=90, hold_frames=150):
    import cv2

    rng = random.Random(seed)
    fonts = [font] if font else find_fonts()
    if not fonts:
        raise FileNotFoundError("한글 글꼴을 찾지 못했습니다. Linux·Colab: apt-get install fonts-nanum")
    if not font:   # 학습에 쓴 글꼴 중 굵은 고딕 계열을 우선. 나눔스퀘어 굵게(크기 96)에서 모인 자모가 가장 안정적으로 검출됨
        train = split_fonts(fonts)["train"]
        order = ("nanumsquareb", "malgunbd", "nanumgothicbold", "malgun", "nanumbarungothicbold")
        rank = lambda f: next((i for i, k in enumerate(order) if Path(f).name.lower().startswith(k)), len(order))
        preferred = sorted((f for f in train if rank(f) < len(order)), key=rank)
        font = (preferred or train or fonts)[0]

    # 자음 그림의 실제 크기를 재서 글자 배치 간격을 정합니다 (글꼴마다 자모 그림 크기가 다름).
    g = float(np.median([max(render(c, font, size, (0, 0, 0)).size) for c in "ㄱㄴㄹㅁㅅㅇㅈㅎ"]))
    layouts = [word_layout(w, width, height, g) for w in words]
    # 자모 물체: 단어마다 필요한 자모 + 어느 단어에도 속하지 않는 방해 자모
    objects = []
    for wi, layout in enumerate(layouts):
        for char, tx, ty in layout:
            objects.append(dict(char=char, word=wi, target=(tx, ty)))
    for char in DISTRACTORS[:max(2, 18 - len(objects))]:
        objects.append(dict(char=char, word=None, target=None))
    palette = [(200, 40, 40), (30, 90, 200), (20, 140, 70), (150, 60, 170), (220, 120, 20), (20, 20, 20), (240, 240, 240)]
    for obj in objects:
        obj["sprite"] = render(obj["char"], font, size, rng.choice(palette))
        obj["x"], obj["y"] = rng.uniform(80, width - 80), rng.uniform(80, height - 80)
        speed, heading = rng.uniform(1.0, 2.6), rng.uniform(0, 2 * math.pi)
        obj["vx"], obj["vy"] = speed * math.cos(heading), speed * math.sin(heading)
        obj["amp"], obj["omega"], obj["phase"] = rng.uniform(10, 30), rng.uniform(0.03, 0.07), rng.uniform(0, 6.3)

    # 단계 계획
    phases, frame = [], 0
    phases.append(dict(kind="떠다님", start=frame, end=frame + float_frames - 1, word=None))
    frame += float_frames
    for wi, word in enumerate(words):
        phases.append(dict(kind="모임", start=frame, end=frame + gather_frames - 1, word=wi))
        frame += gather_frames
        phases.append(dict(kind="멈춤", start=frame, end=frame + hold_frames - 1, word=wi))
        frame += hold_frames
        phases.append(dict(kind="흩어짐", start=frame, end=frame + float_frames - 1, word=None))
        frame += float_frames
    total = frame

    band = 0.22 * height   # 모임·멈춤 동안 비켜난 자모가 떠다니는 위·아래 띠 높이
    bg = background(width, height, rng)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"영상을 쓸 수 없습니다: {output}")
    phase_of = {f: p for p in phases for f in range(p["start"], p["end"] + 1)}
    for f in range(total):
        phase = phase_of[f]
        image = bg.copy()
        for obj in objects:
            active = phase["word"] is not None and obj["word"] == phase["word"]
            if active:
                if phase["kind"] == "모임":
                    if f == phase["start"]:
                        obj["from"] = (obj["x"], obj["y"], obj["angle"])
                    t = ease((f - phase["start"] + 1) / (phase["end"] - phase["start"] + 1))
                    fx, fy, fa = obj["from"]
                    obj["x"] = fx + (obj["target"][0] - fx) * t
                    obj["y"] = fy + (obj["target"][1] - fy) * t
                    obj["angle"] = fa * (1 - t)
                else:   # 멈춤: 아주 살짝만 흔들림
                    obj["x"], obj["y"] = obj["target"]
                    obj["angle"] = 1.5 * math.sin(obj["phase"] + 0.1 * f)
            else:
                # 떠다님. 모임·멈춤 동안에는 위·아래 띠 안에서만 떠다녀 단어와 섞이지 않게 합니다.
                obj["x"] += obj["vx"]
                obj["y"] += obj["vy"]
                if obj["x"] < 40 or obj["x"] > width - 40:
                    obj["vx"] = -obj["vx"]
                    obj["x"] = min(max(obj["x"], 40), width - 40)
                if phase["word"] is not None:
                    top_band = obj["y"] < height / 2
                    low, high = (40, band) if top_band else (height - band, height - 40)
                    obj["y"] += (min(max(obj["y"], low), high) - obj["y"]) * 0.08   # 띠 쪽으로 천천히 이동
                    if (obj["y"] <= low and obj["vy"] < 0) or (obj["y"] >= high and obj["vy"] > 0):
                        obj["vy"] = -obj["vy"]
                elif obj["y"] < 40 or obj["y"] > height - 40:
                    obj["vy"] = -obj["vy"]
                    obj["y"] = min(max(obj["y"], 40), height - 40)
                obj["angle"] = obj["amp"] * math.sin(obj["phase"] + obj["omega"] * f)
            sprite = rotate(obj["sprite"], obj["angle"])
            paste(image, sprite, int(round(obj["x"] - sprite.width / 2)), int(round(obj["y"] - sprite.height / 2)))
        writer.write(image)
    writer.release()

    info = dict(words=list(words), fps=fps, frames=total, width=width, height=height, font=Path(font).name,
                size=size, glyph=g, seed=seed,
                phases=[dict(p, word=words[p["word"]] if p["word"] is not None else None) for p in phases],
                note="'멈춤' 단계에는 그 단어가 화면 가운데에 완성된 모양으로 있습니다. 확장 기능은 같은 단어가 "
                     "8프레임 이어져야 표시하므로 '모임' 끝무렵부터 표시될 수 있습니다.")
    Path(output).with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--words", nargs="+", default=["한글", "객체추적"])
    parser.add_argument("--output", default=str(ROOT / "hangul_demo.mp4"))
    parser.add_argument("--size", type=int, default=96, help="자모 글꼴 크기 (픽셀)")
    parser.add_argument("--font", help="글꼴 파일 (기본: 학습에 쓴 고딕 계열 글꼴)")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(f"이미 있습니다: {args.output}")
    result = make_demo(args.words, args.output, size=args.size, seed=args.seed, font=args.font)
    print(f"{args.output}: {result['frames']}프레임 ({result['frames'] / result['fps']:.0f}초), 글꼴 {result['font']}")
    for p in result["phases"]:
        print(f"  {p['start']:>4}~{p['end']:<4} {p['kind']} {p['word'] or ''}")
