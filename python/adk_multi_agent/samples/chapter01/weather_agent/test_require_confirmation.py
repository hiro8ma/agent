"""FunctionTool の require_confirmation が何をできて何ができないかを測る。

自作の hitl.Gate と同じ機構（request_confirmation / tool_confirmation）を
使うが、判断が bool しかない。3 値との差がどこに出るかを確かめる。
"""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents.llm_agent import Agent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

CALLED: list[dict] = []


def refund(amount: int) -> dict:
    """返金を実行する。

    Args:
        amount: 返金額（円）
    """
    CALLED.append({"amount": amount})
    return {"status": "success", "amount": amount}


def over_1000(amount: int = 0, **_) -> bool:
    """1000 円を超えたら確認を求める。"""
    return amount > 1000


class CallOnce(BaseLlm):
    """1 回だけツールを呼び、その後は黙るモデル。"""

    amount: int = 0
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
                                name="refund", args={"amount": self.amount}
                            )
                        )
                    ],
                )
            )
            return
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="完了")])
        )


async def _run(amount: int) -> list[dict]:
    CALLED.clear()
    agent = Agent(
        name="refund_agent",
        model=CallOnce(model="scripted", amount=amount),
        instruction="返金してください。",
        tools=[FunctionTool(func=refund, require_confirmation=over_1000)],
    )
    runner = InMemoryRunner(agent=agent, app_name="rc")
    await runner.session_service.create_session(
        app_name="rc", user_id="u1", session_id="s1"
    )
    responses = []
    async for event in runner.run_async(
        user_id="u1",
        session_id="s1",
        new_message=types.Content(role="user", parts=[types.Part(text="返金して")]),
    ):
        for part in (event.content.parts if event.content else []) or []:
            if part.function_response is not None:
                responses.append(part.function_response.response)
    return responses


async def test_asks_when_callable_returns_true():
    """しきい値を超えたら確認を求め、ツール本体は動かないことを見る。"""
    responses = await _run(5000)
    joined = str(responses)
    assert "confirmation" in joined, joined
    assert CALLED == [], f"確認前にツールが動いた: {CALLED}"


async def test_runs_directly_when_callable_returns_false():
    """しきい値以下なら確認を挟まず実行することを見る。

    何でも聞くなら仕組みとして機能していない。
    """
    await _run(10)
    assert CALLED == [{"amount": 10}], f"実行されていない: {CALLED}"


async def test_cannot_deny_outright():
    """どれだけ高額でも「聞かずに拒む」ができないことを見る。

    require_confirmation は bool なので、Ask と Deny を区別できない。
    Gate の Threshold は deny を超えた額を人に聞かずに拒むが、
    こちらは同じ額でも人に投げる。承認されれば通ってしまう。
    """
    responses = await _run(99_999_999)
    joined = str(responses)
    assert "confirmation" in joined, joined
    assert "rejected" not in joined, "拒否できるなら Gate は不要になる"


def test_two_mechanisms_not_four():
    """承認の書き口は 4 つでも、下の機構は 2 つであることを見る。

    LongRunningFunctionTool は FunctionTool に is_long_running を
    立てただけ。request_input もその実体になる。
    どちらの系に乗るかで、止まる場所と再開の経路が変わる。
    """
    from google.adk.tools import (
        FunctionTool,
        LongRunningFunctionTool,
        request_input,
    )

    def noop(x: str) -> dict:
        """何もしない。

        Args:
            x: 値
        """
        return {}

    # LongRunning 系は FunctionTool の派生で、フラグが立つ
    assert issubclass(LongRunningFunctionTool, FunctionTool)
    assert LongRunningFunctionTool(func=noop).is_long_running is True
    assert FunctionTool(func=noop).is_long_running is False

    # ADK が提供する request_input も LongRunning 系
    assert isinstance(request_input, LongRunningFunctionTool)

    # ToolConfirmation 系はフラグを立てない
    confirmed = FunctionTool(func=noop, require_confirmation=True)
    assert confirmed.is_long_running is False, (
        "require_confirmation は LongRunning 系ではない"
    )
