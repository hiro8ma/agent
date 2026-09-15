"""日程を組むエージェント。

前段の結果は State の鍵で読む。instruction を関数にすると
`{spot_research}` のような波括弧の置換は行われないので、
ctx.state から自分で取り出して文字列に埋める。

鍵が無いときも落とさず、「調査結果なし」と書いて進める。
並列の 1 本だけが失敗したときに、全体を止めないため。
"""

from __future__ import annotations

from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext

from .researchers import MODEL, RESTAURANT_KEY, SPOT_KEY, TRANSPORT_KEY

SCHEDULE_KEY = "schedule"


def planner_instruction(ctx: ReadonlyContext) -> str:
    """State の調査結果 3 つを読み、instruction を組み立てる。"""
    spots = ctx.state.get(SPOT_KEY, "観光スポットの調査結果なし")
    restaurants = ctx.state.get(RESTAURANT_KEY, "レストランの調査結果なし")
    transport = ctx.state.get(TRANSPORT_KEY, "交通手段の調査結果なし")
    return (
        "あなたは日程の担当です。"
        "下の調査結果だけを使って、日ごとの日程表を作ります。"
        "1 日を午前と昼と午後と夕方に分け、各枠にスポットか食事を 1 つ置きます。"
        "スポットの所要時間と移動を考え、1 日に詰め込みすぎません。"
        "調査結果に無いスポットや店を足しません。\n\n"
        f"観光スポットの調査結果:\n{spots}\n\n"
        f"レストランの調査結果:\n{restaurants}\n\n"
        f"交通手段の調査結果:\n{transport}"
    )


schedule_planner = Agent(
    name="schedule_planner",
    model=MODEL,
    description="3 つの調査結果を統合して日程表を作るエージェント",
    instruction=planner_instruction,
    output_key=SCHEDULE_KEY,
)
