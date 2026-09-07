"""天気と観光の流れを Graph-based Workflow で固定する。

weather_agent は同じ処理を tools として持ち、呼ぶ順序をモデルに任せる。
こちらは edges で順序を固定する。どちらが正しいかではなく、
決める主体が違う。

    START → resolve_city → fetch_weather → fetch_sightseeing → compose

関数ノードの引数は 2 通りに束縛される。

    node_input という名前   前段の出力がそのまま入る
    それ以外の名前          ctx.state から引かれる

state から引く場合、上流が値を書いていなければ ValueError で落ちる。
Agent ノードなら output_key、関数ノードなら ctx.state への書き込みが要る。

ノードを 1 つ外すと、その場で ValueError になる。黙って劣化しない。
"""

from __future__ import annotations

from google.adk import Context, Event, Workflow

# 都市名の別名。weather_agent と重複するが、共有しない。
# adk web は各エージェントのディレクトリを独立したパッケージとして読むため、
# 兄弟パッケージからの import は ModuleNotFoundError になる。
_ALIASES = {
    "東京": "tokyo", "とうきょう": "tokyo",
    "大阪": "osaka", "おおさか": "osaka",
    "札幌": "sapporo", "さっぽろ": "sapporo",
    "福岡": "fukuoka", "ふくおか": "fukuoka",
}


def normalize(city: str) -> str:
    raw = city.strip()
    return _ALIASES.get(raw, raw.lower())

_WEATHER = {
    "tokyo": ("晴れ", 28),
    "osaka": ("曇り", 30),
    "sapporo": ("雨", 21),
    "fukuoka": ("晴れ", 31),
}

_SIGHTS = {
    "tokyo": ["浅草寺", "東京タワー", "明治神宮"],
    "osaka": ["大阪城", "道頓堀", "通天閣"],
    "sapporo": ["大通公園", "時計台", "藻岩山"],
    "fukuoka": ["太宰府天満宮", "櫛田神社", "福岡城跡"],
}


def resolve_city(node_input: str, ctx: Context) -> str:
    """入力から都市を決め、後段が読めるよう state へ置く。

    表示用の元の表記も残す。鍵は英字小文字なので、
    そのまま応答へ出すと「sapporo は雨です」になる。
    """
    city = normalize(node_input)
    ctx.state["city"] = city
    ctx.state["display"] = node_input.strip()
    return city


def fetch_weather(city: str, ctx: Context) -> str:
    """state の city で天気を引く。"""
    if city not in _WEATHER:
        ctx.state["weather"] = ""
        return ""
    condition, temp = _WEATHER[city]
    ctx.state["weather"] = f"{condition}、気温 {temp} 度"
    return ctx.state["weather"]


def fetch_sightseeing(city: str, ctx: Context) -> str:
    """state の city で観光地を引く。"""
    spots = _SIGHTS.get(city, [])
    ctx.state["sights"] = "、".join(spots)
    return ctx.state["sights"]


def compose(display: str, weather: str, sights: str) -> Event:
    """3 つの state を 1 つの応答へまとめる。

    ここまで LLM を 1 度も呼んでいない。順序も分岐もコードで決まる。
    """
    if not weather:
        return Event(output=f"{display} の情報は登録されていません。")
    return Event(output=f"{display} は{weather}です。{sights} を回れます。")


root_agent = Workflow(
    name="weather_workflow",
    edges=[("START", resolve_city, fetch_weather, fetch_sightseeing, compose)],
)
