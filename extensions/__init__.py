"""확장 기능 모음 (과제 「확장 문제」).

확장 기능 하나 = 파일 하나입니다. 각 파일은 :class:`extensions.base.Extension`을 상속한 클래스를 하나 정의하고,
아래 ``REGISTRY``에 한 줄로 등록합니다. 파이프라인·GUI·결과 파일은 이 등록 정보만 보고 동작하므로,
새 확장 기능을 만들 때 ``pipeline.py``·``main.py``를 고칠 필요가 없습니다.

| key          | 파일             | 담당                    |
|--------------|------------------|-------------------------|
| ``hangul``     | ``hangul.py``     | Pdint (Claude)          |
| ``camouflage`` | ``camouflage.py`` | kumohyunho (ChatGPT)    |

자세한 규칙: ``docs/COLLABORATION.md``
"""
from importlib import import_module

# key → "모듈:클래스". 모듈은 실제로 쓸 때만 불러옵니다 (OCR 등 무거운 패키지를 GUI 시작 때 읽지 않도록).
REGISTRY = {
    "hangul": "extensions.hangul:HangulJamoExtension",
    "camouflage": "extensions.camouflage:CamouflageExtension",
}


def load(key):
    """``key``에 해당하는 확장 기능 클래스를 돌려줍니다."""
    if key not in REGISTRY:
        raise ValueError(f"알 수 없는 확장 기능: {key} (가능: {', '.join(REGISTRY)})")
    module, name = REGISTRY[key].split(":")
    return getattr(import_module(module), name)


def available_extensions():
    """GUI 목록용: ``[(key, 표시 이름, 사용할 수 없는 이유 또는 None), ...]``."""
    items = []
    for key in REGISTRY:
        try:
            cls = load(key)
            items.append((key, cls.label, cls.check()))
        except Exception as error:  # 모듈 오류가 GUI 전체를 막지 않도록 이유로만 표시합니다.
            items.append((key, key, f"불러오기 실패: {error}"))
    return items


def source_tags():
    """결과 화면에서 ID 앞에 붙이는 짧은 표시 (``{key: tag}``)."""
    tags = {}
    for key in REGISTRY:
        try:
            tags[key] = load(key).tag
        except Exception:
            tags[key] = key[:2]
    return tags
