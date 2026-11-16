"""확장 기능(한글 자모)용 합성 데이터 생성: 배경 위에 떠다니는 한글 자음·모음.  설계: ``docs/HANGUL.md``

자음·모음 **하나하나가 객체**입니다 (40클래스: 자음 19 + 모음 21, ``extensions/hangul.py``의 ``JAMO``).
떠다니는 자모의 추적 정답(ID)이 있는 공개 데이터가 없으므로, 정답을 알고 있는 합성 데이터를 직접 만듭니다.

1. 검출 학습용 이미지 (YOLO 형식, 40클래스) — ``images/<split>``, ``labels/<split>``
   * 자모 하나 = 박스 하나. 색·크기·글꼴·외곽선·바탕 판·**회전(기본 ±45°)**을 무작위로 바꿉니다.
   * ``annotations/<split>.jsonl``: 박스별 자모와 회전 각도 (회전 각도별 정확도 분석용)
2. 추적 평가용 "떠다니는 자모" 영상 (MOTChallenge 형식) — ``tracking/<split>/<seq>``
   * 자모 여러 개가 떠다니며 천천히 회전하고, 서로 교차하고, 가림막에 가려지고, 화면 밖으로 나갔다 다시 들어옵니다.
   * **같은 자모가 여러 개** 함께 나옵니다 (모양이 같은 객체끼리 ID가 섞이지 않는지 확인).
   * ``gt/gt.txt``: ``frame,id,left,top,width,height,1,1,visibility`` (MOT17과 같은 형식, 기존 지표 코드와 호환)
   * ``gt/objects.json``: ID별 자모(클래스 이름·글자), ``gt/angles.csv``: ``frame,id,angle`` (프레임별 회전 각도)

회전 범위를 ±45°로 제한하는 이유: 자모는 돌리면 다른 글자가 됩니다 (ㅏ를 180° 돌리면 ㅓ, ㅗ↔ㅜ, ㄱ↔ㄴ, ㅡ를 90° 돌리면 ㅣ).
±45° 안에서는 원래 글자를 구분할 수 있습니다. 같은 이유로 학습할 때 좌우 반전 증강을 꺼야 합니다 (``--fliplr 0``).

train / val / test는 처음부터 따로 생성하므로 겹치지 않습니다. 글꼴이 5개 이상이면 약 20%를 **test 전용 글꼴**로
떼어 두어, 학습에 쓰지 않은 글꼴에서도 동작하는지 확인합니다 (``splits.json``에 기록).

배경은 ``sample.mp4`` 프레임 + 무늬 배경을 기본으로 쓰고, ``--backgrounds``로 사진 폴더(예: COCO val2017)를 주면 함께 씁니다.

예)
    python hangul_dataset.py --output datasets/hangul
    python hangul_dataset.py --output datasets/hangul --backgrounds datasets/raw/coco/val2017 --train 8000
    python train_models.py hangul --data datasets/hangul/dataset.yaml --fliplr 0 --epochs 50 --install

글꼴: Windows는 맑은 고딕·굴림·바탕, Linux·Colab은 ``apt-get install fonts-nanum``(나눔 글꼴)을 자동으로 찾습니다.
다른 글꼴은 ``--fonts <폴더>``로 추가합니다.
"""
import argparse
import hashlib
import json
import math
import platform
import random
from pathlib import Path

import numpy as np

from extensions.hangul import CHAR, CLASS_NAMES

ROOT = Path(__file__).resolve().parent

FONT_DIRS = {
    "Windows": [Path("C:/Windows/Fonts"), Path.home() / "AppData/Local/Microsoft/Windows/Fonts"],
    "Darwin": [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts"],
    "Linux": [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
              Path.home() / ".local/share/fonts"],
}
KOREAN_FONT_HINTS = ("nanum", "malgun", "gulim", "batang", "dotum", "gungsuh", "applesdgothic", "applemyungjo",
                     "notosanskr", "notoserifkr", "notosanscjk", "notoserifcjk", "pretendard", "spoqa", "hmkm", "hy")


# ---------------------------------------------------------------------- 글꼴
def supports_jamo(path):
    """40개 자모가 모두 비어 있지 않고 서로 다르게 그려지는 글꼴만 사용합니다."""
    from PIL import ImageFont

    try:
        font = ImageFont.truetype(str(path), 32)
        shapes = set()
        for char in CHAR.values():
            mask = font.getmask(char)
            if not mask.getbbox():
                return False
            shapes.add((mask.size, bytes(mask)))
        return len(shapes) == len(CHAR)
    except Exception:
        return False


def find_fonts(extra_dirs=()):
    """자모를 그릴 수 있는 한글 글꼴 경로 목록 (이름순, 중복 제거)."""
    extra = [Path(d) for d in extra_dirs]
    candidates = []
    for folder in extra + FONT_DIRS.get(platform.system(), FONT_DIRS["Linux"]):
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.suffix.lower() not in (".ttf", ".otf", ".ttc"):
                continue
            # 추가 폴더는 모두 시험하고, 시스템 폴더는 한글 글꼴로 보이는 이름만 시험합니다 (수백 개라 느림).
            if folder in extra or any(h in path.name.lower() for h in KOREAN_FONT_HINTS):
                candidates.append(path)
    fonts = sorted({p.resolve() for p in candidates if supports_jamo(p)}, key=lambda p: p.name.lower())
    return [str(p) for p in fonts]


def split_fonts(fonts):
    """글꼴이 5개 이상이면 약 20%를 test 전용으로 떼어 둡니다 (파일 이름 해시 기준, 항상 같은 결과)."""
    if len(fonts) < 5:
        return {"train": fonts, "val": fonts, "test": fonts}
    held = [f for f in fonts if int(hashlib.md5(Path(f).name.encode()).hexdigest(), 16) % 5 == 0]
    if not held or len(held) == len(fonts):
        held = fonts[-max(1, len(fonts) // 5):]
    seen = [f for f in fonts if f not in held]
    return {"train": seen, "val": seen, "test": held}


# ---------------------------------------------------------------------- 배경
class Backgrounds:
    """``sample.mp4`` 프레임, 사진 폴더, 무늬 배경에서 무작위로 고릅니다."""

    def __init__(self, rng, folders=(), video=ROOT / "sample.mp4", video_frames=60):
        import cv2

        self.rng = rng
        self.images = []
        for folder in folders:
            self.images += sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self.frames = []
        if Path(video).is_file():
            capture = cv2.VideoCapture(str(video))
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            for index in np.linspace(0, total - 1, min(video_frames, total)).astype(int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if ok:
                    self.frames.append(frame)
            capture.release()

    def pattern(self, width, height):
        rng = self.rng
        kind = rng.choice(["gradient", "noise", "blocks", "flat"])
        base = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
        if kind == "gradient":
            other = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
            t = np.linspace(0, 1, width if rng.random() < 0.5 else height, dtype=np.float32)
            line = base + (other - base) * t[:, None]
            image = np.repeat(line[None], height, 0) if len(t) == width else np.repeat(line[:, None], width, 1)
        elif kind == "noise":
            image = base + np.random.default_rng(rng.randint(0, 2**31)).normal(0, 40, (height, width, 3))
        else:
            image = np.tile(base, (height, width, 1))
            for _ in range(rng.randint(3, 10) if kind == "blocks" else 0):
                x, y = rng.randint(0, width - 1), rng.randint(0, height - 1)
                image[y:y + rng.randint(20, height), x:x + rng.randint(20, width)] = [rng.randint(0, 255) for _ in range(3)]
        return np.clip(image, 0, 255).astype(np.uint8)

    def get(self, width, height):
        import cv2

        rng = self.rng
        sources = (["photo"] * 3 if self.images else []) + (["video"] * 2 if self.frames else []) + ["pattern"]
        source = rng.choice(sources)
        if source == "pattern":
            return self.pattern(width, height)
        if source == "photo":
            image = cv2.imdecode(np.fromfile(rng.choice(self.images), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return self.pattern(width, height)
        else:
            image = rng.choice(self.frames)
        scale = max(width / image.shape[1], height / image.shape[0]) * rng.uniform(1.0, 1.4)
        image = cv2.resize(image, (int(image.shape[1] * scale) + 1, int(image.shape[0] * scale) + 1))
        x = rng.randint(0, image.shape[1] - width)
        y = rng.randint(0, image.shape[0] - height)
        return image[y:y + height, x:x + width].copy()


# ---------------------------------------------------------------------- 자모 그리기
def _crop(layer, margin=2):
    """투명하지 않은 부분만 남기고 ``margin``픽셀 여유를 둡니다 (ㅡ, ㅣ처럼 얇은 자모의 박스가 너무 얇아지지 않게)."""
    alpha = np.asarray(layer)[:, :, 3]
    ys, xs = np.nonzero(alpha > 20)
    if len(xs) == 0:
        return None
    return layer.crop((max(0, xs.min() - margin), max(0, ys.min() - margin),
                       min(layer.width, xs.max() + 1 + margin), min(layer.height, ys.max() + 1 + margin)))


def render_jamo(char, font_path, size, rng, background_luma):
    """자모 하나를 똑바로 선 RGBA 그림으로 그립니다. 배경 밝기와 대비가 나는 색을 고릅니다."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, size)
    stroke = rng.choice([0, 0, 0, 1, 2]) if size >= 28 else 0
    left, top, right, bottom = font.getbbox(char, stroke_width=stroke)
    pad = max(3, size // 5)
    side = max(right - left, bottom - top) + 2 * pad
    layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    def color(bright):
        if rng.random() < 0.45:  # 선명한 색 (떠다니는 글자 놀이 화면처럼)
            hue = np.uint8([[[rng.randint(0, 179), rng.randint(140, 255), rng.randint(200, 255) if bright else rng.randint(60, 130)]]])
            import cv2

            b, g, r = cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)[0, 0]
            return int(r), int(g), int(b)
        value = rng.randint(190, 255) if bright else rng.randint(0, 70)
        return tuple(int(np.clip(value + rng.randint(-50, 50), 0, 255)) for _ in range(3))

    plate = rng.random() < 0.25  # 풍선·카드처럼 바탕 판 위에 쓴 자모
    if plate:
        plate_bright = rng.random() < 0.5
        shape = [0, 0, side - 1, side - 1]
        if rng.random() < 0.5:
            draw.ellipse(shape, fill=color(plate_bright) + (255,))
        else:
            draw.rounded_rectangle(shape, radius=pad, fill=color(plate_bright) + (255,))
        fill = color(not plate_bright)
    else:
        fill = color(background_luma < 128)
    outline = (0, 0, 0) if sum(fill) / 3 >= 128 else (255, 255, 255)
    x = (side - (right - left)) // 2 - left
    y = (side - (bottom - top)) // 2 - top
    draw.text((x, y), char, font=font, fill=fill + (255,), stroke_width=stroke, stroke_fill=outline + (255,))
    return _crop(layer, margin=0 if plate else 2)


def rotate(sprite, angle):
    """``angle``도(반시계 방향) 돌린 그림. 박스는 돌린 뒤의 글자를 감싸는 축 정렬 사각형입니다."""
    from PIL import Image

    if abs(angle) < 0.5:
        return sprite
    return _crop(sprite.rotate(angle, resample=Image.BICUBIC, expand=True), margin=1)


def paste(image, sprite, x, y):
    """BGR ``image``에 RGBA ``sprite``를 (x, y)에 합성합니다. 화면 밖으로 나간 부분은 잘립니다."""
    rgba = np.asarray(sprite)
    h, w = rgba.shape[:2]
    H, W = image.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    part = rgba[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = part[:, :, 3:4].astype(np.float32) / 255.0
    region = image[y1:y2, x1:x2].astype(np.float32)
    image[y1:y2, x1:x2] = (region * (1 - alpha) + part[:, :, 2::-1] * alpha).astype(np.uint8)


def covered_ratio(box, boxes):
    """``box`` 넓이 중 ``boxes``와 겹치는 부분의 최대 비율."""
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    worst = 0.0
    for a1, b1, a2, b2 in boxes:
        iw, ih = min(x2, a2) - max(x1, a1), min(y2, b2) - max(y1, b1)
        if iw > 0 and ih > 0:
            worst = max(worst, iw * ih / min(area, (a2 - a1) * (b2 - b1)))
    return worst


# ---------------------------------------------------------------------- 1. 검출 학습 이미지
def make_image(rng, backgrounds, fonts, rotation):
    width, height = rng.choice([(640, 640), (640, 480), (960, 540), (800, 600)])
    image = backgrounds.get(width, height)
    luma = image.mean(axis=2)
    items, boxes = [], []
    for _ in range(rng.randint(3, 14)):
        name = rng.choice(CLASS_NAMES)
        size = int(rng.choice([rng.uniform(22, 40), rng.uniform(40, 80), rng.uniform(80, 140)]))
        angle = rng.uniform(-rotation, rotation) if rng.random() < 0.85 else 0.0
        for _attempt in range(8):
            x, y = rng.randint(0, width - 20), rng.randint(0, height - 20)
            sprite = render_jamo(CHAR[name], rng.choice(fonts), size, rng, float(luma[y:y + size, x:x + size].mean()))
            sprite = rotate(sprite, angle) if sprite is not None else None
            if sprite is None or sprite.width >= width or sprite.height >= height:
                size = max(20, size // 2)
                continue
            x, y = min(x, width - sprite.width), min(y, height - sprite.height)
            box = (x, y, x + sprite.width, y + sprite.height)
            if covered_ratio(box, boxes) > 0.15:  # 살짝 겹치는 것까지만 허용
                continue
            paste(image, sprite, x, y)
            boxes.append(box)
            items.append({"box": list(box), "class": name, "char": CHAR[name], "angle": round(angle, 1)})
            break
    if rng.random() < 0.3:  # 흐림 등 실제 영상 품질
        import cv2

        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.3, 1.2))
    return image, items


def write_detection_split(rng, backgrounds, fonts, output, split, count, rotation):
    import cv2

    (output / "images" / split).mkdir(parents=True, exist_ok=True)
    (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "annotations").mkdir(exist_ok=True)
    per_class = dict.fromkeys(CLASS_NAMES, 0)
    with (output / "annotations" / f"{split}.jsonl").open("w", encoding="utf-8") as notes:
        for index in range(count):
            image, items = make_image(rng, backgrounds, fonts, rotation)
            name = f"{split}_{index:06d}"
            cv2.imwrite(str(output / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
            height, width = image.shape[:2]
            lines = []
            for item in items:
                x1, y1, x2, y2 = item["box"]
                lines.append(f"{CLASS_NAMES.index(item['class'])} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                             f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n")
                per_class[item["class"]] += 1
            (output / "labels" / split / f"{name}.txt").write_text("".join(lines), encoding="utf-8")
            notes.write(json.dumps({"image": f"images/{split}/{name}.jpg", "items": items}, ensure_ascii=False) + "\n")
            if (index + 1) % 200 == 0 or index + 1 == count:
                print(f"\r{split}: {index + 1}/{count}", end="", flush=True)
    print()
    counts = list(per_class.values())
    return {"images": count, "boxes": sum(counts), "min_per_class": min(counts), "max_per_class": max(counts)}


# ---------------------------------------------------------------------- 2. 떠다니는 자모 영상
def write_sequence(rng, backgrounds, fonts, folder, name, length, rotation, fps=30, width=960, height=540):
    """자모들이 떠다니며 회전·교차·가림·화면 이탈 후 재진입하는 영상과 MOT 형식 정답을 만듭니다."""
    import configparser

    import cv2

    background = backgrounds.get(width, height)
    pan = (rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2))  # 배경이 천천히 흐르는 효과
    luma = float(background.mean())
    count = rng.randint(8, 14)
    names = [rng.choice(CLASS_NAMES) for _ in range(count - 3)]
    names += [rng.choice(names) for _ in range(3)]  # 같은 자모가 적어도 몇 쌍은 함께 나오게
    objects = []
    for track_id, class_name in enumerate(names, start=1):
        sprite = render_jamo(CHAR[class_name], rng.choice(fonts), rng.randint(40, 90), rng, luma)
        if sprite is None:
            continue
        speed = rng.uniform(1.0, 5.0)
        heading = rng.uniform(0, 2 * math.pi)
        objects.append(dict(id=track_id, cls=class_name, sprite=sprite,
                            cx=rng.uniform(60, width - 60), cy=rng.uniform(60, height - 60),
                            vx=speed * math.cos(heading), vy=speed * math.sin(heading),
                            # 회전: ±amplitude 안에서 천천히 흔들립니다.
                            amplitude=rng.uniform(0, rotation), omega=rng.uniform(0.02, 0.08),
                            phase=rng.uniform(0, 2 * math.pi),
                            leave=rng.random() < 0.4, outside=0, wait=rng.randint(20, 50)))
    # 가림막: 화면을 가로지르는 기둥 (가림 상황)
    bar_w = rng.randint(60, 140)
    bar = dict(x=-bar_w if rng.random() < 0.5 else width, speed=rng.uniform(2, 5), w=bar_w,
               color=[rng.randint(0, 255) for _ in range(3)])
    bar["speed"] *= 1 if bar["x"] < 0 else -1

    (folder / "img1").mkdir(parents=True, exist_ok=True)
    (folder / "gt").mkdir(exist_ok=True)
    gt_lines, angle_lines = [], ["frame,id,angle\n"]
    for frame in range(1, length + 1):
        shift = (int(pan[0] * frame) % width, int(pan[1] * frame) % height)
        image = np.roll(np.roll(background, shift[0], axis=1), shift[1], axis=0).copy()
        bx1 = int(bar["x"])
        cover = np.zeros((height, width), dtype=bool)
        cover[:, max(0, bx1):max(0, min(width, bx1 + bar["w"]))] = True
        for obj in objects:
            if obj["outside"]:  # 화면 밖에서 기다렸다가 반대 방향으로 다시 들어옵니다.
                obj["outside"] -= 1
                if obj["outside"] == 0:
                    obj["vx"], obj["vy"] = -obj["vx"], -obj["vy"]
                continue
            obj["cx"] += obj["vx"]
            obj["cy"] += obj["vy"]
            angle = obj["amplitude"] * math.sin(obj["phase"] + obj["omega"] * frame)
            sprite = rotate(obj["sprite"], angle)
            w, h = sprite.size
            for axis, half, limit in (("cx", w / 2, width), ("cy", h / 2, height)):
                velocity = "v" + axis[1]
                low, high = obj[axis] - half, obj[axis] + half
                # 바깥쪽으로 움직일 때만 처리합니다 (다시 들어오는 중에는 그대로 둡니다).
                if (low < 0 and obj[velocity] < 0) or (high > limit and obj[velocity] > 0):
                    if obj["leave"]:
                        if high < 0 or low > limit:
                            obj["outside"] = obj["wait"]
                            obj["leave"] = rng.random() < 0.3
                    else:
                        obj[velocity] = -obj[velocity]
            x, y = int(round(obj["cx"] - w / 2)), int(round(obj["cy"] - h / 2))
            paste(image, sprite, x, y)
            x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            visibility = (x2 - x1) * (y2 - y1) / (w * h) * (1 - cover[y1:y2, x1:x2].mean())
            gt_lines.append(f"{frame},{obj['id']},{x1},{y1},{x2 - x1},{y2 - y1},1,1,{visibility:.3f}\n")
            angle_lines.append(f"{frame},{obj['id']},{angle:.1f}\n")
        if bx1 + bar["w"] > 0 and bx1 < width:
            image[:, max(0, bx1):min(width, bx1 + bar["w"])] = bar["color"]
        bar["x"] += bar["speed"]
        if bar["x"] > width + 200 or bar["x"] < -bar["w"] - 200:
            bar["speed"] = -bar["speed"]
        cv2.imwrite(str(folder / "img1" / f"{frame:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (folder / "gt" / "gt.txt").write_text("".join(gt_lines), encoding="utf-8")
    (folder / "gt" / "angles.csv").write_text("".join(angle_lines), encoding="utf-8")
    (folder / "gt" / "objects.json").write_text(json.dumps(
        {str(o["id"]): {"class": o["cls"], "char": CHAR[o["cls"]]} for o in objects}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    info = configparser.ConfigParser()
    info.optionxform = str
    info["Sequence"] = dict(name=name, imDir="img1", frameRate=str(fps), seqLength=str(length),
                            imWidth=str(width), imHeight=str(height), imExt=".jpg")
    with (folder / "seqinfo.ini").open("w", encoding="utf-8") as handle:
        info.write(handle, space_around_delimiters=False)
    chars = [CHAR[o["cls"]] for o in objects]
    return {"frames": length, "ids": len(objects), "boxes": len(gt_lines),
            "duplicated_jamo": sorted({c for c in chars if chars.count(c) > 1})}


# ---------------------------------------------------------------------- 실행
def main(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"출력 폴더가 비어 있지 않습니다: {output}")
    fonts = find_fonts(args.fonts or [])
    if not fonts:
        raise FileNotFoundError("자모를 그릴 한글 글꼴을 찾지 못했습니다. Linux·Colab: apt-get install fonts-nanum, "
                                "또는 --fonts <글꼴 폴더>를 지정하세요.")
    font_splits = split_fonts(fonts)
    print(f"한글 글꼴 {len(fonts)}개 (test 전용 {len(font_splits['test']) if len(fonts) >= 5 else 0}개), 회전 ±{args.rotation:g}°")
    rng = random.Random(args.seed)
    backgrounds = Backgrounds(rng, args.backgrounds or [])
    output.mkdir(parents=True, exist_ok=True)

    summary = {"detection": {}, "tracking": {}}
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        summary["detection"][split] = write_detection_split(
            rng, backgrounds, font_splits[split], output, split, count, args.rotation)
    for split in ("val", "test"):
        for index in range(1, args.sequences + 1):
            name = f"jamo-{index:02d}"
            summary["tracking"].setdefault(split, {})[name] = write_sequence(
                rng, backgrounds, font_splits[split], output / "tracking" / split / name, name, args.length,
                args.rotation)
            print(f"{split}/{name}: {summary['tracking'][split][name]}")

    listing = ", ".join(f"{i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (output / "dataset.yaml").write_text(
        f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
        f"train: images/train\nval: images/val\ntest: images/test\nnames: {{{listing}}}\n", encoding="utf-8")
    (output / "splits.json").write_text(json.dumps(dict(
        kind="hangul-jamo-synthetic", seed=args.seed, rotation=args.rotation, classes=CLASS_NAMES,
        chars=[CHAR[n] for n in CLASS_NAMES], counts=summary,
        fonts={k: [Path(f).name for f in v] for k, v in font_splits.items()},
        backgrounds=[str(Path(b)) for b in (args.backgrounds or [])] + ["sample.mp4", "무늬 배경"],
        note="train/val/test는 따로 생성되어 겹치지 않음. 글꼴 5개 이상이면 test는 학습에 쓰지 않은 글꼴만 사용. "
             "학습 시 좌우 반전 증강을 끌 것 (--fliplr 0)."),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(output / "dataset.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--output", required=True, help="예: datasets/hangul")
    parser.add_argument("--train", type=int, default=6000, help="학습 이미지 수 (이미지당 자모 3~14개)")
    parser.add_argument("--val", type=int, default=600)
    parser.add_argument("--test", type=int, default=600)
    parser.add_argument("--sequences", type=int, default=4, help="val·test 각각의 떠다니는 자모 영상 수")
    parser.add_argument("--length", type=int, default=300, help="영상 길이 (프레임, 30fps)")
    parser.add_argument("--rotation", type=float, default=45.0, help="자모 회전 범위 ±도 (0이면 회전 없음)")
    parser.add_argument("--backgrounds", nargs="*", help="배경 사진 폴더 (선택, 예: datasets/raw/coco/val2017)")
    parser.add_argument("--fonts", nargs="*", help="추가 글꼴 폴더 (선택)")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
