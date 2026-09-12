"""調査フェーズ。3 つの専門エージェントを並列で動かす。

並列の子は同じ State を共有する。output_key が重なると、後から書いた方
だけが残り、他は黙って消える。呼び出しの費用は払う。
そのため、組み立て時に鍵の重複を落とす。
"""

from __future__ import annotations

from google.adk import Agent
from google.adk.agents import ParallelAgent

from ..tools import restaurant_search_tool, spot_search_tool, transport_search_tool

MODEL = "gemini-3.5-flash"

SPOT_KEY = "spot_research"
RESTAURANT_KEY = "restaurant_research"
TRANSPORT_KEY = "transport_research"

spot_researcher = Agent(
    name="spot_researcher",
    model=MODEL,
    description="旅行先の観光スポットを調べ、候補を絞って報告するエージェント",
    instruction=(
        "あなたは観光スポットの担当です。"
        "旅行先と好みを読み取り、search_tourist_spots を 1 回だけ呼びます。"
        "結果の name と category と duration_hours と entry_fee_yen をそのまま使い、"
        "候補を 5 件以内で挙げます。"
        "ツールが error を返したら、その都市は登録されていないことだけを伝えます。"
        "ツールの結果に無いスポットを足しません。"
    ),
    tools=[spot_search_tool],
    output_key=SPOT_KEY,
)

restaurant_researcher = Agent(
    name="restaurant_researcher",
    model=MODEL,
    description="旅行先の食事の候補を予算内で調べて報告するエージェント",
    instruction=(
        "あなたは食事の担当です。"
        "旅行先と料理の好みと 1 人あたりの予算を読み取り、search_restaurants を 1 回だけ呼びます。"
        "予算が示されていなければ 0 を渡します。"
        "結果の name と cuisine と budget_per_person_yen をそのまま使い、候補を 5 件以内で挙げます。"
        "ツールが error を返したら、条件に合う店が無いことだけを伝えます。"
        "ツールの結果に無い店を足しません。"
    ),
    tools=[restaurant_search_tool],
    output_key=RESTAURANT_KEY,
)

transport_researcher = Agent(
    name="transport_researcher",
    model=MODEL,
    description="出発地から旅行先までの交通手段を調べて報告するエージェント",
    instruction=(
        "あなたは交通手段の担当です。"
        "出発地と目的地と出発日を読み取り、search_transport を 1 回だけ呼びます。"
        "出発地が示されていなければ東京として扱います。"
        "日付が示されていなければ 2026-10-03 を使います。"
        "結果の mode と duration_minutes と price_yen をそのまま使い、往復の費用も示します。"
        "ツールが error を返したら、その経路は登録されていないことだけを伝えます。"
    ),
    tools=[transport_search_tool],
    output_key=TRANSPORT_KEY,
)

_RESEARCHERS = [spot_researcher, restaurant_researcher, transport_researcher]

_keys = [a.output_key for a in _RESEARCHERS]
if len(set(_keys)) != len(_keys):
    raise ValueError(f"output_key が重複している: {_keys}")

# ParallelAgent は v2.2.0 で非推奨。生成時に DeprecationWarning が出る。
research_phase = ParallelAgent(
    name="research_phase",
    description="観光スポットと食事と交通手段を並列で調べるフェーズ",
    sub_agents=_RESEARCHERS,
)
