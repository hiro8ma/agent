"""教材の、入力した文で承認する HITL の実装を実際に回す。モデルは台本で置き換える。"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk.agents import Agent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from samples.approval import text_approval
from samples.approval.state_gate import EXECUTED, transfer_funds


class Script(BaseLlm):
    """ターンごとに決めた金額で送金を呼び、ツールの応答を受けたら答える。"""

    amounts: list[int] = []
    model_calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.model_calls += 1
        last = llm_request.contents[-1]
        if any(p.function_response for p in last.parts or []) or not self.amounts:
            part = types.Part(text="処理しました")
        else:
            args = {"amount": self.amounts.pop(0), "to": "999-0001"}
            part = types.Part(
                function_call=types.FunctionCall(name="transfer_funds", args=args)
            )
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


@pytest.fixture
async def runner() -> InMemoryRunner:
    EXECUTED.clear()
    agent = Agent(
        name="hitl_agent",
        model=Script(model="gemini-3.8-flash"),
        instruction="x",
        tools=[transfer_funds],
        before_model_callback=text_approval.handle_approval_input,
        before_tool_callback=text_approval.before_tool_callback,
    )
    r = InMemoryRunner(agent=agent, app_name="hitl")
    await r.session_service.create_session(
        app_name="hitl", user_id="alice", session_id="s1"
    )
    return r


async def say(
    runner: InMemoryRunner, text: str, amounts: list[int] | None = None
) -> tuple[list, str]:
    runner.agent.model.amounts = list(amounts or [])
    responses, last_text = [], ""
    message = types.Content(role="user", parts=[types.Part(text=text)])
    async for ev in runner.run_async(
        user_id="alice", session_id="s1", new_message=message
    ):
        for p in ev.content.parts if ev.content else []:
            if p.function_response:
                responses.append(p.function_response.response)
            if p.text:
                last_text = p.text
    return responses, last_text


async def test_requester_approves_own_request_by_typing(runner: InMemoryRunner) -> None:
    """依頼した本人が ID を入力すれば承認になる。承認者は分かれていない。"""
    [pending], _ = await say(runner, "150 万円送金して", [1_500_000])
    request_id = pending["request_id"]
    _, text = await say(runner, f"承認: {request_id}")
    assert text.startswith("承認されました")


async def test_says_executed_but_nothing_is_executed(runner: InMemoryRunner) -> None:
    """承認の直後の応答は「実行します」だが、モデルを呼ばないのでツールは実行されない。"""
    [pending], _ = await say(runner, "150 万円送金して", [1_500_000])
    calls_before = runner.agent.model.model_calls
    responses, text = await say(runner, f"承認: {pending['request_id']}")

    assert "実行します" in text
    assert responses == []
    assert EXECUTED == []
    assert runner.agent.model.model_calls == calls_before


async def test_next_call_with_another_amount_uses_the_approval(
    runner: InMemoryRunner,
) -> None:
    """承認のフラグはツール単位なので、次にモデルが別の額で呼んでも通る。"""
    [pending], _ = await say(runner, "150 万円送金して", [1_500_000])
    await say(runner, f"承認: {pending['request_id']}")
    responses, _ = await say(runner, "お願いします", [9_000_000])

    assert responses[0]["status"] == "sent"
    assert EXECUTED == [{"amount": 9_000_000, "to": "999-0001"}]


async def test_second_request_overwrites_the_first(runner: InMemoryRunner) -> None:
    """承認待ちは 1 枠しかない。2 件目を依頼すると、1 件目の ID では承認できなくなる。"""
    [first], _ = await say(runner, "150 万円送金して", [1_500_000])
    [second], _ = await say(runner, "200 万円も送金して", [2_000_000])
    _, text = await say(runner, f"承認: {first['request_id']}")

    assert first["request_id"] != second["request_id"]
    assert not text.startswith("承認されました")
