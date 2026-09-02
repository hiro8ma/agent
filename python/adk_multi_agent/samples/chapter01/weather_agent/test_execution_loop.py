"""実行ループを API キー無しで流す検査。

台本どおりに応答するモデルを差し込み、観測 → 理解 → 計画 → 行動 → 評価
の 1 往復を観測する。モデルの賢さは測れないが、ツールが呼ばれ、
結果が State に残り、応答に統合されるまでの配線は測れる。
"""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from samples.chapter01.weather_agent.agent import root_agent
from samples.chapter01.weather_agent.tools import LAST_CITY_KEY


class ScriptedModel(BaseLlm):
    """台本の順に応答を返すモデル。"""

    turns: list[types.Content] = []
    calls: int = 0
    seen: list[str] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        if llm_request.tools_dict:
            self.seen = self.seen + list(llm_request.tools_dict)
        i = self.calls
        self.calls = i + 1
        if i >= len(self.turns):
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="台本の終わり")]
                )
            )
            return
        yield LlmResponse(content=self.turns[i])


def call(name: str, args: dict) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
    )


def text(s: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=s)])


@pytest.mark.asyncio
async def test_execution_loop_calls_both_tools_and_integrates():
    """天気を引き、観光を引き、統合して答えるかを見る。"""
    model = ScriptedModel(
        model="scripted",
        turns=[
            call("get_weather", {"city": "東京"}),
            call("get_sightseeing", {"city": "東京"}),
            text("東京は晴れ、気温 28 度です。屋外の浅草寺がおすすめです。"),
        ],
    )
    original = root_agent.model
    root_agent.model = model
    try:
        runner = InMemoryRunner(agent=root_agent, app_name="ch01_loop")
        await runner.session_service.create_session(
            app_name="ch01_loop", user_id="u1", session_id="s1"
        )

        tool_results: list[str] = []
        payloads: list[str] = []
        final = ""
        async for event in runner.run_async(
            user_id="u1",
            session_id="s1",
            new_message=types.Content(
                role="user", parts=[types.Part(text="東京に旅行に行きたいんだけど")]
            ),
        ):
            if not event.content or not event.content.parts:
                continue
            for part in event.content.parts:
                if part.function_response is not None:
                    tool_results.append(part.function_response.name)
                    payloads.append(str(part.function_response.response))
                if part.text:
                    final = part.text

        # 行動: 2 つのツールがこの順で実行されたか
        assert tool_results == ["get_weather", "get_sightseeing"], tool_results

        # 理解: モデルに 2 つのツールが見えていたか
        for name in ("get_weather", "get_sightseeing"):
            assert name in model.seen, f"モデルに {name} が渡っていない"

        # 行動の中身: ツールの戻り値に実データが入っているか。
        # 名前だけ見ると、未登録のツールを呼んだ場合も通ってしまう。
        joined = " ".join(payloads)
        for want in ("晴れ", "浅草寺"):
            assert want in joined, f"ツールの戻り値に {want!r} が無い: {joined}"

        # 台本の最終応答が届いたか。中身は台本なので経路の確認にとどまる。
        assert final

        # 記憶: 直近の都市が残ったか
        session = await runner.session_service.get_session(
            app_name="ch01_loop", user_id="u1", session_id="s1"
        )
        assert session.state.get(LAST_CITY_KEY) == "tokyo", dict(session.state)
    finally:
        root_agent.model = original
