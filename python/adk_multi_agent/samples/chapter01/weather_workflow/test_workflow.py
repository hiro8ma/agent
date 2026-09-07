"""順序をコードで固定した場合と、モデルに任せた場合の差を測る。

同じ「天気と観光を両方引いて 1 つの応答にする」処理を 2 通りで作り、
モデルが片方のツールを飛ばしたときに何が起きるかを比べる。
"""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from samples.chapter01.weather_agent.agent import root_agent as tools_agent
from samples.chapter01.weather_workflow.workflow import root_agent as workflow_agent


class SkipsSightseeing(BaseLlm):
    """天気だけ呼んで観光を飛ばすモデル。

    実際のモデルも、指示に書いてあってもツールを飛ばすことがある。
    その状況を再現する。
    """

    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        i = self.calls
        self.calls = i + 1
        if i == 0:
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="get_weather", args={"city": "東京"}
                            )
                        )
                    ],
                )
            )
            return
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text="東京は晴れ、気温 28 度です。")]
            )
        )


async def _collect(agent, app: str, text: str) -> tuple[list[str], list[str]]:
    runner = InMemoryRunner(agent=agent, app_name=app)
    await runner.session_service.create_session(
        app_name=app, user_id="u1", session_id="s1"
    )
    tools_called: list[str] = []
    outputs: list[str] = []
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        if getattr(event, "output", None) is not None:
            outputs.append(str(event.output))
        for part in (event.content.parts if event.content else []) or []:
            if part.function_response is not None:
                tools_called.append(part.function_response.name)
            if part.text:
                outputs.append(part.text)
    return tools_called, outputs


async def test_workflow_runs_every_step_regardless_of_any_model():
    """Workflow 版は観光を必ず引くことを見る。

    LLM を 1 度も呼ばないので、モデルの気分に左右されない。
    """
    _, outputs = await _collect(workflow_agent, "wf", "東京")
    joined = " ".join(outputs)
    assert "晴れ" in joined, joined
    assert "浅草寺" in joined, f"観光が落ちている: {joined}"


async def test_tools_version_loses_a_step_when_the_model_skips_it():
    """対照。ツール版はモデルが飛ばすと観光が落ちることを見る。

    Instruction には「get_weather と get_sightseeing を呼び」と
    書いてあるが、守られる保証はない。
    """
    original = tools_agent.model
    tools_agent.model = SkipsSightseeing(model="skips")
    try:
        called, outputs = await _collect(tools_agent, "tl", "東京")
    finally:
        tools_agent.model = original

    assert called == ["get_weather"], f"台本どおりでない: {called}"
    joined = " ".join(outputs)
    assert "浅草寺" not in joined, "観光が出てしまった。対照が成立していない"


async def test_workflow_reports_unknown_city_without_crashing():
    """未登録の都市でも落ちずに伝えることを見る。"""
    _, outputs = await _collect(workflow_agent, "wf2", "那覇")
    joined = " ".join(outputs)
    assert "登録されていません" in joined, joined
