"""予算を計算し、最終レポートを構造化して返すエージェント。

output_schema を付けると、最終出力が TravelPlan として検証される。
v2.2.0 では output_schema とツールの併用も許される。
ツールは思考の途中で使え、形が強制されるのは最終出力だけ。
ここではツールを持たせず、State に載っている調査結果と日程だけを使う。
"""

from __future__ import annotations

from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext

from ..schemas import TravelPlan
from .planner import SCHEDULE_KEY
from .researchers import MODEL, RESTAURANT_KEY, SPOT_KEY, TRANSPORT_KEY

REPORT_KEY = "travel_plan"


def reporter_instruction(ctx: ReadonlyContext) -> str:
    """State の日程と調査結果を読み、instruction を組み立てる。"""
    schedule = ctx.state.get(SCHEDULE_KEY, "日程表なし")
    spots = ctx.state.get(SPOT_KEY, "観光スポットの調査結果なし")
    restaurants = ctx.state.get(RESTAURANT_KEY, "レストランの調査結果なし")
    transport = ctx.state.get(TRANSPORT_KEY, "交通手段の調査結果なし")
    return (
        "あなたは予算の担当です。"
        "下の日程表と調査結果から概算の予算を出し、TravelPlan の形で返します。"
        "交通費は往復で数えます。食費は日程に置いた食事の合計です。"
        "入場料は日程に置いたスポットの合計です。"
        "total_yen は 3 項目の和と一致させます。"
        "金額が調査結果から分からない項目は 0 にして、tips にその旨を書きます。\n\n"
        f"日程表:\n{schedule}\n\n"
        f"観光スポットの調査結果:\n{spots}\n\n"
        f"レストランの調査結果:\n{restaurants}\n\n"
        f"交通手段の調査結果:\n{transport}"
    )


budget_reporter = Agent(
    name="budget_reporter",
    model=MODEL,
    description="予算を計算し、TravelPlan の形で最終レポートを返すエージェント",
    instruction=reporter_instruction,
    output_schema=TravelPlan,
    output_key=REPORT_KEY,
)
