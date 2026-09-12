"""実機の Gemini で 1 往復だけ流す。

台本モデルの検査は配線しか見られない。実際のモデルが 3 本の調査を並列で
回し、最後に TravelPlan の形で返すかは、ここで確かめる。

実行:
    GOOGLE_API_KEY=... uv run python -m samples.chapter02.travel_planner.live_run

無料枠は 1 分 5 回なので、5 エージェントで上限に当たることがある。
その場合は 429 が返る。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

from .agent import root_agent
from .agents import REPORT_KEY, RESTAURANT_KEY, SCHEDULE_KEY, SPOT_KEY, TRANSPORT_KEY

PROMPT = (
    "東京から京都へ 2 泊 3 日で行きます。歴史と自然が好きです。"
    "食事は 1 人 5000 円まで。出発は 2026-10-03 です。"
)

APP = "travel_planner_live"

# 無料枠はモデルごとに 1 分 5 回。このパイプラインは並列調査で 6 回、日程と予算で 2 回呼ぶので、
# 何もしないと途中で 429 になる。宣言側は教材どおりのモデル名のままにして、実機で流すここだけ待たせる。
RETRY = types.HttpRetryOptions(
    attempts=8,
    initial_delay=20,
    max_delay=90,
    exp_base=1.5,
    http_status_codes=[429, 503],
)


def _attach_retrying_models(agent) -> None:
    """パイプライン内の全エージェントに、429 を待つモデルを差し込む。"""
    if isinstance(getattr(agent, "model", None), str) and agent.model:
        agent.model = Gemini(model=agent.model, retry_options=RETRY)
    for child in getattr(agent, "sub_agents", None) or []:
        _attach_retrying_models(child)


async def main() -> int:
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("GOOGLE_API_KEY か GEMINI_API_KEY を設定する", file=sys.stderr)
        return 1

    _attach_retrying_models(root_agent)

    runner = InMemoryRunner(agent=root_agent, app_name=APP)
    await runner.session_service.create_session(
        app_name=APP, user_id="u1", session_id="s1"
    )

    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(role="user", parts=[types.Part(text=PROMPT)]),
    ):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            if part.function_call is not None:
                print(
                    f"[{event.author}] call {part.function_call.name} {part.function_call.args}"
                )
            if part.function_response is not None:
                print(f"[{event.author}] result {part.function_response.name}")
            if part.text:
                print(f"[{event.author}] {part.text[:200]}")

    session = await runner.session_service.get_session(
        app_name=APP, user_id="u1", session_id="s1"
    )
    state = dict(session.state)
    print("\n--- state の鍵 ---")
    for key in (SPOT_KEY, RESTAURANT_KEY, TRANSPORT_KEY, SCHEDULE_KEY, REPORT_KEY):
        print(f"{key}: {'あり' if key in state else 'なし'}")

    report = state.get(REPORT_KEY)
    if report is None:
        print("TravelPlan が State に無い", file=sys.stderr)
        return 1
    parsed = json.loads(report) if isinstance(report, str) else report
    print("\n--- TravelPlan ---")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
