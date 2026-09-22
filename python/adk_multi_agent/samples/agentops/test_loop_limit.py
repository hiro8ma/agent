"""同じツールを呼び続けるエージェントが、ADK の上限でどこまで回るかを確かめる。モデルは台本で置き換える。"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from samples.agentops.loop_guard import RepeatedToolCallGuard


class Looping(BaseLlm):
    """何を受けても同じツールを同じ引数で呼ぶ。"""

    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        self.calls += 1
        call = types.FunctionCall(name="search", args={"query": "在庫"})
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(function_call=call)])
        )


def search(query: str) -> dict:
    """在庫を検索する。"""
    del query
    return {"status": "no_result"}


async def run(
    run_config: RunConfig | None, plugins=None
) -> tuple[Looping, Exception | None]:
    model = Looping(model="gemini-3.8-flash")
    agent = Agent(name="loop", model=model, instruction="x", tools=[search])
    runner = InMemoryRunner(agent=agent, app_name="ops", plugins=plugins or [])
    await runner.session_service.create_session(
        app_name="ops", user_id="u", session_id="s"
    )
    message = types.Content(role="user", parts=[types.Part(text="在庫を調べて")])
    error = None
    try:
        async for _ in runner.run_async(
            user_id="u", session_id="s", new_message=message, run_config=run_config
        ):
            pass
    except Exception as e:  # noqa: BLE001
        error = e
    return model, error


async def test_default_limit_allows_500_llm_calls() -> None:
    """既定の max_llm_calls は 500 で、同じツールの呼び出しが 500 回続いてから止まる。"""
    model, error = await run(None)
    assert model.calls == 500
    assert "500" in str(error)


async def test_lower_limit_stops_early() -> None:
    model, error = await run(RunConfig(max_llm_calls=5))
    assert model.calls == 5
    assert error is not None


@pytest.mark.parametrize("limit", [3, 5])
async def test_repeated_call_guard_stops_same_call(limit: int) -> None:
    """同じツールを同じ引数で limit 回呼んだら、次の呼び出しを止めて打ち切る。"""
    model, error = await run(
        RunConfig(max_llm_calls=100), [RepeatedToolCallGuard(limit=limit)]
    )
    assert error is None
    # limit 回実行し、limit+1 回目のツールの呼び出しを止める。Invocation を終わらせるので、次のモデルの呼び出しは無い。
    assert model.calls == limit + 1
