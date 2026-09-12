"""第 2 章 ハンズオン: 旅行プランナー。

    research_phase（並列）→ schedule_planner → budget_reporter

前半は 3 本の調査を並列に走らせ、後半は日程と予算を順に作る。
State の受け渡しは output_key と動的 instruction で行う。

    spot_research / restaurant_research / transport_research
        → schedule → travel_plan（TravelPlan で構造化）

SequentialAgent と ParallelAgent は v2.2.0 で非推奨で、生成時に
DeprecationWarning が出る。案内されている代替は Workflow。
ここで使い続けるのは、教材の構成をそのまま確かめるため。

実行方法:
    adk run samples/chapter02/travel_planner
    uv run python -m samples.chapter02.travel_planner.live_run
"""

from __future__ import annotations

from google.adk.agents import SequentialAgent

from .agents import budget_reporter, research_phase, schedule_planner

root_agent = SequentialAgent(
    name="travel_planner",
    description="調査を並列に行い、日程表と予算レポートを順に作るパイプライン",
    sub_agents=[research_phase, schedule_planner, budget_reporter],
)
