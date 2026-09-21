"""承認を State に置く設計の抜け道を固定する。モデルは呼ばない。"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk.agents import Agent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from samples.approval import state_gate
from samples.approval.state_gate import EXECUTED, require_approval, transfer_funds


class Calls(BaseLlm):
    """台本の送金を 1 回ずつ呼び、呼び終えたら答える。"""

    amounts: list[int] = []
    step: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        last = self.step >= len(self.amounts)
        if last:
            part = types.Part(text="処理しました")
        else:
            args = {"amount": self.amounts[self.step], "to": "999-0001"}
            part = types.Part(
                function_call=types.FunctionCall(name="transfer_funds", args=args)
            )
        self.step += 1
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


async def turn(
    runner: InMemoryRunner, amounts: list[int], state_delta: dict | None = None
) -> list:
    runner.agent.model.amounts, runner.agent.model.step = amounts, 0
    message = types.Content(role="user", parts=[types.Part(text="送金して")])
    responses = []
    async for ev in runner.run_async(
        user_id="u1", session_id="s1", new_message=message, state_delta=state_delta
    ):
        for p in ev.content.parts if ev.content else []:
            if p.function_response:
                responses.append(p.function_response.response)
    return responses


@pytest.fixture
async def runner() -> InMemoryRunner:
    EXECUTED.clear()
    agent = Agent(
        name="treasury",
        model=Calls(model="gemini-3.8-flash"),
        instruction="x",
        tools=[transfer_funds],
        before_tool_callback=require_approval,
    )
    r = InMemoryRunner(agent=agent, app_name="treasury")
    await r.session_service.create_session(
        app_name="treasury", user_id="u1", session_id="s1"
    )
    return r


async def test_gate_stops_large_transfer(runner: InMemoryRunner) -> None:
    [response] = await turn(runner, [1_500_000])
    assert response["status"] == "pending_approval"
    assert EXECUTED == []


async def test_client_can_approve_itself_through_state_delta(
    runner: InMemoryRunner,
) -> None:
    """REST の /run は state_delta を受け、そのまま State に書く。利用者が承認済みの印を書ける。"""
    [response] = await turn(
        runner, [1_500_000], state_delta={"approval:transfer_funds": "approved"}
    )
    assert response["status"] == "sent"
    assert EXECUTED == [{"amount": 1_500_000, "to": "999-0001"}]


async def test_approval_is_not_bound_to_the_amount(runner: InMemoryRunner) -> None:
    """人が 150 万円の承認待ちを見て承認したあと、同じ印で 1,000 万円が通る。"""
    await turn(runner, [1_500_000])
    session = await runner.session_service.get_session(
        app_name="treasury", user_id="u1", session_id="s1"
    )
    assert session.state["approval:pending"]["args"]["amount"] == 1_500_000

    # 承認の画面が State に承認済みを書く、という教材の流れ。
    await turn(
        runner, [], state_delta={state_gate.approval_key("transfer_funds"): "approved"}
    )
    responses = await turn(runner, [10_000_000, 30_000_000])
    assert [r["status"] for r in responses] == ["sent", "sent"]
    assert [e["amount"] for e in EXECUTED] == [10_000_000, 30_000_000]
