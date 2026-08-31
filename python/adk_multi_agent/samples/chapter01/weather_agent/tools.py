"""天気と観光のツール。

都市名は日本語と英語のどちらでも受ける。片方だけにすると、
表記の変換をモデルに任せることになり失敗点が増える。
"""

from __future__ import annotations

from google.adk.tools import ToolContext

_WEATHER = {
    "tokyo": {"condition": "晴れ", "temperature": 28, "humidity": 55},
    "osaka": {"condition": "曇り", "temperature": 30, "humidity": 70},
    "sapporo": {"condition": "雨", "temperature": 21, "humidity": 85},
    "fukuoka": {"condition": "晴れ", "temperature": 31, "humidity": 65},
}

_SIGHTSEEING = {
    "tokyo": {"spots": ["浅草寺", "東京タワー", "明治神宮"], "season": "春（桜の季節）"},
    "osaka": {"spots": ["大阪城", "道頓堀", "通天閣"], "season": "秋"},
    "sapporo": {"spots": ["大通公園", "時計台", "藻岩山"], "season": "冬（雪まつり）"},
    "fukuoka": {"spots": ["太宰府天満宮", "櫛田神社", "福岡城跡"], "season": "春"},
}

_ALIASES = {
    "東京": "tokyo", "とうきょう": "tokyo",
    "大阪": "osaka", "おおさか": "osaka",
    "札幌": "sapporo", "さっぽろ": "sapporo",
    "福岡": "fukuoka", "ふくおか": "fukuoka",
}

# 直近に問い合わせた都市を置く鍵。user: を付けて別セッションでも残す。
LAST_CITY_KEY = "user:last_city"


def normalize(city: str) -> str:
    raw = city.strip()
    return _ALIASES.get(raw, raw.lower())


def _remember(tool_context: ToolContext | None, city: str) -> None:
    if tool_context is not None:
        tool_context.state[LAST_CITY_KEY] = city


def get_weather(city: str, tool_context: ToolContext | None = None) -> dict:
    """指定された都市の現在の天気を取得する。

    Args:
        city: 都市名。日本語（東京）と英語（tokyo）のどちらでもよい。
        tool_context: ADK が注入する実行時コンテキスト。

    Returns:
        status が success なら気温・天気・湿度、error なら error_message。
    """
    key = normalize(city)
    if key not in _WEATHER:
        return {"status": "error", "error_message": f"{city} の天気情報は登録されていない"}
    _remember(tool_context, key)
    d = _WEATHER[key]
    return {
        "status": "success",
        "city": key,
        "condition": d["condition"],
        "temperature": d["temperature"],
        "humidity": d["humidity"],
    }


def get_sightseeing(city: str, tool_context: ToolContext | None = None) -> dict:
    """指定された都市の観光情報を取得する。

    Args:
        city: 都市名。日本語（東京）と英語（tokyo）のどちらでもよい。
        tool_context: ADK が注入する実行時コンテキスト。

    Returns:
        status が success なら観光スポットと見頃、error なら error_message。
    """
    key = normalize(city)
    if key not in _SIGHTSEEING:
        return {"status": "error", "error_message": f"{city} の観光情報は登録されていない"}
    _remember(tool_context, key)
    d = _SIGHTSEEING[key]
    return {
        "status": "success",
        "city": key,
        "spots": d["spots"],
        "recommended_season": d["season"],
    }
