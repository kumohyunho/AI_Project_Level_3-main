"""확장 기능(한글 인식)용 합성 데이터 생성: 배경 위에 한글 글자를 그려 넣습니다.

실제 사진 데이터(예: AI Hub 한글 이미지)는 신청·내려받기가 필요하고, 움직이는 글자의 **추적 정답(ID)**은 어디에도 없습니다.
그래서 정답을 알고 있는 합성 데이터를 직접 만듭니다.

1. 검출 학습용 이미지 (YOLO 형식, 클래스 1개 ``text``) — ``images/<split>``, ``labels/<split>``
   * 글자 한 줄 = 박스 하나. 색·크기·글꼴·외곽선·간판 배경·기울기를 무작위로 바꿉니다.
   * ``annotations/<split>.jsonl``: 박스별 정답 문자열 (OCR 정확도 평가용)
2. 추적 평가용 "떠다니는 한글" 영상 (MOTChallenge 형식) — ``tracking/<split>/<seq>``
   * 글자 여러 개가 움직이며 서로 교차하고, 가림막에 가려지고, 화면 밖으로 나갔다 다시 들어옵니다.
   * ``gt/gt.txt``: ``frame,id,left,top,width,height,1,1,visibility`` (MOT17과 같은 형식)
   * ``gt/texts.json``: ID별 정답 문자열

train / val / test는 처음부터 따로 생성하므로 겹치지 않습니다. 글꼴이 5개 이상이면 약 20%를 **test 전용 글꼴**로
떼어 두어, 학습에 쓰지 않은 글꼴에서도 동작하는지 확인합니다 (``splits.json``에 기록).

배경은 ``sample.mp4`` 프레임 + 무늬 배경을 기본으로 쓰고, ``--backgrounds``로 사진 폴더(예: COCO val2017)를 주면 함께 씁니다.

예)
    python hangul_dataset.py --output datasets/hangul
    python hangul_dataset.py --output datasets/hangul --backgrounds datasets/raw/coco/val2017 --train 6000

글꼴: Windows는 맑은 고딕·굴림·바탕, Linux·Colab은 ``apt-get install fonts-nanum``(나눔 글꼴)을 자동으로 찾습니다.
다른 글꼴은 ``--fonts <폴더>``로 추가합니다.
"""
import argparse
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["text"]

# 간판·안내문·자막에 자주 나오는 낱말. 무작위 음절 조합과 섞어 씁니다.
WORDS = """
출구 입구 화장실 엘리베이터 계단 주차장 편의점 약국 병원 은행 우체국 경찰서 소방서 학교 도서관 박물관 미술관
지하철 버스 정류장 택시 승강장 매표소 안내 환영합니다 감사합니다 어서오세요 영업중 준비중 휴무 할인 세일 신상품
커피 빵집 식당 분식 치킨 피자 김밥 라면 떡볶이 냉면 국밥 삼겹살 카페 음료 아이스크림 과일 채소 정육점 꽃집 서점
미용실 세탁소 사진관 안경점 휴대폰 노래방 피시방 헬스장 수영장 공원 놀이터 운동장 체육관 강당 교실 기숙사 식당
금연 주의 위험 정지 서행 진입금지 우회전 좌회전 직진 비상구 소화기 관계자외출입금지 조용히 손씻기 마스크착용
대한민국 서울 부산 대구 인천 광주 대전 울산 세종 경기 강원 충청 전라 경상 제주 구미 금오공과대학교
인공지능 객체추적 컴퓨터비전 딥러닝 머신러닝 데이터 모델 학습 검증 테스트 추적 검출 인식 영상 카메라 화면
사랑 행복 희망 친구 가족 하늘 바다 구름 바람 나무 꽃 봄 여름 가을 겨울 아침 저녁 오늘 내일 시작 끝
""".split()
NUMBER_WORDS = ["1번 출구", "2번 출구", "3층", "지하 1층", "10% 할인", "24시간", "B2 주차장", "5분 거리", "100m 앞"]
COMMON_SYLLABLES = list("가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호"
                        "구누두루무부수우주추쿠투푸후그느드르므브스으즈츠크트프흐기니디리미비시이지치키티피히"
                        "개내대래매배새애재채해게네데레메베세에제체한국인민중대학생일사람시간말년전정상회")

FONT_DIRS = {
    "Windows": [Path("C:/Windows/Fonts"), Path.home() / "AppData/Local/Microsoft/Windows/Fonts"],
    "Darwin": [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library/Fonts"],
    "Linux": [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
              Path.home() / ".local/share/fonts"],
}
KOREAN_FONT_HINTS = ("nanum", "malgun", "gulim", "batang", "dotum", "gungsuh", "applesdgothic", "applemyungjo",
                     "notosanskr", "notoserifkr", "notosanscjk", "notoserifcjk", "pretendard", "spoqa", "hmkm", "hy")


# ---------------------------------------------------------------------- 글꼴
def supports_hangul(path):
    """글꼴에 한글 글리프가 있는지: '한'과 '글'이 서로 다르고 비어 있지 않게 그려지면 사용합니다."""
    from PIL import ImageFont

    try:
        font = ImageFont.truetype(str(path), 32)
        first, second = font.getmask("한"), font.getmask("글")
        if not first.getbbox() or not second.getbbox():
            return False
        return bytes(first) != bytes(second)
    except Exception:
        return False


def find_fonts(extra_dirs=()):
    """한글 글꼴 경로 목록 (이름순, 중복 제거)."""
    candidates = []
    for folder in list(extra_dirs) + FONT_DIRS.get(platform.system(), FONT_DIRS["Linux"]):
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.suffix.lower() not in (".ttf", ".otf", ".ttc"):
                continue
            # 추가 폴더는 모두 시험하고, 시스템 폴더는 한글 글꼴로 보이는 이름만 시험합니다 (수백 개라 느림).
            if Path(folder) in map(Path, extra_dirs) or any(h in path.name.lower() for h in KOREAN_FONT_HINTS):
                candidates.append(path)
    fonts = sorted({p.resolve() for p in candidates if supports_hangul(p)}, key=lambda p: p.name.lower())
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


# ---------------------------------------------------------------------- 문자열
def random_text(rng):
    roll = rng.random()
    if roll < 0.55:
        return rng.choice(WORDS)
    if roll < 0.7:
        return f"{rng.choice(WORDS)} {rng.choice(WORDS)}"
    if roll < 0.8:
        return rng.choice(NUMBER_WORDS)
    return "".join(rng.choice(COMMON_SYLLABLES) for _ in range(rng.randint(2, 6)))


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
        kind = rng.choice(["gradient", "noise", "blocks"])
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
            for _ in range(rng.randint(3, 10)):
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


# ---------------------------------------------------------------------- 글자 그리기
def render_text(text, font_path, size, rng, background_luma):
    """글자 한 줄을 RGBA 그림으로 그립니다. 배경 밝기와 대비가 나는 색을 고릅니다."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(font_path, size)
    stroke = rng.choice([0, 0, 0, 1, 2]) if size >= 20 else 0
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    pad = max(2, size // 6)
    plate = rng.random() < 0.35  # 간판처럼 단색 판 위에 쓴 글자
    width, height = right - left + 2 * pad, bottom - top + 2 * pad
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    def color(bright):
        value = rng.randint(190, 255) if bright else rng.randint(0, 70)
        tint = [value + rng.randint(-60, 0) if bright else value + rng.randint(0, 60) for _ in range(3)]
        return tuple(int(np.clip(c, 0, 255)) for c in tint)

    if plate:
        plate_bright = rng.random() < 0.5
        draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=pad, fill=color(plate_bright) + (255,))
        fill = color(not plate_bright)
    else:
        fill = color(background_luma < 128)
    outline = color(sum(fill) / 3 < 128)
    draw.text((pad - left, pad - top), text, font=font, fill=fill + (255,),
              stroke_width=stroke, stroke_fill=outline + (255,))
    angle = rng.uniform(-8, 8) if rng.random() < 0.5 else 0.0
    if angle:
        layer = layer.rotate(angle, resample=Image.BICUBIC, expand=True)
    alpha = np.asarray(layer)[:, :, 3]
    ys, xs = np.nonzero(alpha > 20)
    if len(xs) == 0:
        return None
    return layer.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def paste(image, sprite, x, y, clip=None):
    """BGR ``image``에 RGBA ``sprite``를 (x, y)에 합성합니다. 화면 밖으로 나간 부분은 잘립니다."""
    rgba = np.asarray(sprite)
    h, w = rgba.shape[:2]
    H, W = image.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    part = rgba[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = part[:, :, 3:4].astype(np.float32) / 255.0
    if clip is not None:
        alpha = alpha * clip[y1:y2, x1:x2, None]
    region = image[y1:y2, x1:x2].astype(np.float32)
    image[y1:y2, x1:x2] = (region * (1 - alpha) + part[:, :, 2::-1] * alpha).astype(np.uint8)


def overlap(box, boxes):
    x1, y1, x2, y2 = box
    for a1, b1, a2, b2 in boxes:
        iw, ih = min(x2, a2) - max(x1, a1), min(y2, b2) - max(y1, b1)
        if iw > 0 and ih > 0:
            return True
    return False


# ---------------------------------------------------------------------- 1. 검출 학습 이미지
def make_image(rng, backgrounds, fonts):
    width, height = rng.choice([(640, 640), (640, 480), (960, 540), (800, 600)])
    image = backgrounds.get(width, height)
    luma = image.mean(axis=2)
    items, boxes = [], []
    for _ in range(rng.randint(1, 7)):
        text = random_text(rng)
        size = int(rng.choice([rng.uniform(14, 30), rng.uniform(30, 70), rng.uniform(70, 120)]))
        for _attempt in range(10):
            # 그릴 위치의 배경 밝기를 보고 색을 정하므로 위치를 먼저 대략 고릅니다.
            x, y = rng.randint(0, width - 20), rng.randint(0, height - 20)
            sprite = render_text(text, rng.choice(fonts), size, rng, float(luma[y:y + size, x:x + size * 2].mean()))
            if sprite is None or sprite.width >= width or sprite.height >= height:
                size = max(12, size // 2)
                continue
            x, y = min(x, width - sprite.width), min(y, height - sprite.height)
            box = (x, y, x + sprite.width, y + sprite.height)
            if overlap(box, boxes):
                continue
            paste(image, sprite, x, y)
            boxes.append(box)
            items.append({"box": list(box), "text": text})
            break
    if rng.random() < 0.3:  # 흐림·압축 손실 등 실제 영상 품질
        import cv2

        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.3, 1.2))
    return image, items


def write_detection_split(rng, backgrounds, fonts, output, split, count):
    import cv2

    (output / "images" / split).mkdir(parents=True, exist_ok=True)
    (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "annotations").mkdir(exist_ok=True)
    boxes = 0
    with (output / "annotations" / f"{split}.jsonl").open("w", encoding="utf-8") as notes:
        for index in range(count):
            image, items = make_image(rng, backgrounds, fonts)
            name = f"{split}_{index:06d}"
            cv2.imwrite(str(output / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
            height, width = image.shape[:2]
            lines = []
            for item in items:
                x1, y1, x2, y2 = item["box"]
                lines.append(f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                             f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n")
            (output / "labels" / split / f"{name}.txt").write_text("".join(lines), encoding="utf-8")
            notes.write(json.dumps({"image": f"images/{split}/{name}.jpg", "items": items}, ensure_ascii=False) + "\n")
            boxes += len(items)
            if (index + 1) % 200 == 0 or index + 1 == count:
                print(f"\r{split}: {index + 1}/{count}", end="", flush=True)
    print()
    return {"images": count, "boxes": boxes}


# ---------------------------------------------------------------------- 2. 떠다니는 한글 영상
def write_sequence(rng, backgrounds, fonts, folder, name, length, fps=30, width=960, height=540):
    """글자들이 움직이며 교차·가림·화면 이탈 후 재진입하는 영상과 MOT 형식 정답을 만듭니다."""
    import configparser

    import cv2

    background = backgrounds.get(width, height)
    pan = (rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2))  # 배경이 천천히 흐르는 효과
    luma = float(background.mean())
    objects = []
    for track_id in range(1, rng.randint(3, 7) + 1):
        text = random_text(rng)
        sprite = None
        for size in (rng.randint(28, 80), 28):
            sprite = render_text(text, rng.choice(fonts), size, rng, luma)
            if sprite is not None and sprite.width < width * 0.6:
                break
        if sprite is None or sprite.width >= width * 0.8:
            continue
        speed = rng.uniform(1.5, 6.0)
        angle = rng.uniform(0, 2 * np.pi)
        objects.append(dict(id=track_id, text=text, sprite=sprite,
                            x=rng.uniform(0, width - sprite.width), y=rng.uniform(0, height - sprite.height),
                            vx=speed * np.cos(angle), vy=speed * np.sin(angle),
                            leave=rng.random() < 0.5, outside=0, wait=rng.randint(20, 50)))
    # 가림막: 화면을 가로지르는 기둥 (가림 상황)
    bar_w = rng.randint(60, 140)
    bar = dict(x=-bar_w if rng.random() < 0.5 else width, speed=rng.uniform(2, 5), w=bar_w,
               color=[rng.randint(0, 255) for _ in range(3)])
    bar["speed"] *= 1 if bar["x"] < 0 else -1

    (folder / "img1").mkdir(parents=True, exist_ok=True)
    (folder / "gt").mkdir(exist_ok=True)
    gt_lines = []
    for frame in range(1, length + 1):
        shift = (int(pan[0] * frame) % width, int(pan[1] * frame) % height)
        image = np.roll(np.roll(background, shift[0], axis=1), shift[1], axis=0).copy()
        bx1 = int(bar["x"])
        cover = np.zeros((height, width), dtype=bool)
        cover[:, max(0, bx1):max(0, min(width, bx1 + bar["w"]))] = True
        for obj in objects:
            w, h = obj["sprite"].size
            if obj["outside"]:  # 화면 밖에서 기다렸다가 반대 방향으로 다시 들어옵니다.
                obj["outside"] -= 1
                if obj["outside"] == 0:
                    obj["vx"], obj["vy"] = -obj["vx"], -obj["vy"]
                continue
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]
            for axis, size, limit in (("x", w, width), ("y", h, height)):
                velocity = "v" + axis
                # 바깥쪽으로 움직일 때만 처리합니다 (다시 들어오는 중에는 그대로 둡니다).
                if (obj[axis] < 0 and obj[velocity] < 0) or (obj[axis] + size > limit and obj[velocity] > 0):
                    fully_out = obj[axis] + size < 0 or obj[axis] > limit
                    if obj["leave"] and fully_out:
                        obj["outside"] = obj["wait"]
                        obj["leave"] = rng.random() < 0.3
                    elif not obj["leave"]:
                        obj[axis] = float(np.clip(obj[axis], 0, limit - size))
                        obj[velocity] = -obj[velocity]
            x, y = int(round(obj["x"])), int(round(obj["y"]))
            paste(image, obj["sprite"], x, y)
            x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            covered = cover[y1:y2, x1:x2].mean()
            visibility = (x2 - x1) * (y2 - y1) / (w * h) * (1 - covered)
            gt_lines.append(f"{frame},{obj['id']},{x1},{y1},{x2 - x1},{y2 - y1},1,1,{visibility:.3f}\n")
        if bx1 + bar["w"] > 0 and bx1 < width:
            image[:, max(0, bx1):min(width, bx1 + bar["w"])] = bar["color"]
        bar["x"] += bar["speed"]
        if bar["x"] > width + 200 or bar["x"] < -bar["w"] - 200:
            bar["speed"] = -bar["speed"]
        cv2.imwrite(str(folder / "img1" / f"{frame:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (folder / "gt" / "gt.txt").write_text("".join(gt_lines), encoding="utf-8")
    (folder / "gt" / "texts.json").write_text(
        json.dumps({str(o["id"]): o["text"] for o in objects}, ensure_ascii=False, indent=2), encoding="utf-8")
    info = configparser.ConfigParser()
    info.optionxform = str
    info["Sequence"] = dict(name=name, imDir="img1", frameRate=str(fps), seqLength=str(length),
                            imWidth=str(width), imHeight=str(height), imExt=".jpg")
    with (folder / "seqinfo.ini").open("w", encoding="utf-8") as handle:
        info.write(handle, space_around_delimiters=False)
    return {"frames": length, "ids": len(objects), "boxes": len(gt_lines)}


# ---------------------------------------------------------------------- 실행
def main(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"출력 폴더가 비어 있지 않습니다: {output}")
    fonts = find_fonts(args.fonts or [])
    if not fonts:
        raise FileNotFoundError("한글 글꼴을 찾지 못했습니다. Linux·Colab: apt-get install fonts-nanum, "
                                "또는 --fonts <글꼴 폴더>를 지정하세요.")
    font_splits = split_fonts(fonts)
    print(f"한글 글꼴 {len(fonts)}개 (test 전용 {len(font_splits['test']) if len(fonts) >= 5 else 0}개)")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    backgrounds = Backgrounds(rng, args.backgrounds or [])
    output.mkdir(parents=True, exist_ok=True)

    summary = {"detection": {}, "tracking": {}}
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        summary["detection"][split] = write_detection_split(rng, backgrounds, font_splits[split], output, split, count)
    for split in ("val", "test"):
        for index in range(1, args.sequences + 1):
            name = f"hangul-{index:02d}"
            summary["tracking"].setdefault(split, {})[name] = write_sequence(
                rng, backgrounds, font_splits[split], output / "tracking" / split / name, name, args.length)
            print(f"{split}/{name}: {summary['tracking'][split][name]}")

    (output / "dataset.yaml").write_text(
        f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\n"
        "train: images/train\nval: images/val\ntest: images/test\nnames: {0: text}\n", encoding="utf-8")
    (output / "splits.json").write_text(json.dumps(dict(
        kind="hangul-synthetic", seed=args.seed, classes=CLASS_NAMES, counts=summary,
        fonts={k: [Path(f).name for f in v] for k, v in font_splits.items()},
        backgrounds=[str(Path(b)) for b in (args.backgrounds or [])] + ["sample.mp4", "무늬 배경"],
        note="train/val/test는 따로 생성되어 겹치지 않음. 글꼴 5개 이상이면 test는 학습에 쓰지 않은 글꼴만 사용."),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(output / "dataset.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--output", required=True, help="예: datasets/hangul")
    parser.add_argument("--train", type=int, default=3000, help="학습 이미지 수")
    parser.add_argument("--val", type=int, default=500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--sequences", type=int, default=4, help="val·test 각각의 떠다니는 한글 영상 수")
    parser.add_argument("--length", type=int, default=300, help="영상 길이 (프레임, 30fps)")
    parser.add_argument("--backgrounds", nargs="*", help="배경 사진 폴더 (선택, 예: datasets/raw/coco/val2017)")
    parser.add_argument("--fonts", nargs="*", help="추가 글꼴 폴더 (선택)")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
