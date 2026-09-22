"""エスカレーション、実行回数の制限、監査ログの教材の書き方が、State を信じることで何を許すかを確かめる。モデルは台本で置き換える。"""

from __future__ import annotations

from collections import Counter
from collections.abc import AsyncGenerator

from google.adk import Agent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types


class Calls(BaseLlm):
    """最初に決めた数だけ、同じツールを並行に呼び、結果を受けたら終わる。"""

    n: int = 1

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        if llm_request.contents[-1].parts[-1].function_response:
            parts = [types.Part(text="終わりました")]
        else:
            call = types.FunctionCall(name="lookup", args={"q": "x"})
            parts = [types.Part(function_call=call) for _ in range(self.n)]
        yield LlmResponse(content=types.Content(role="model", parts=parts))


def lookup(q: str) -> dict:
    """検索する。"""
    del q
    return {"hit": 1}


async def run(
    agent: Agent, user_id: str = "u", state_delta: dict | None = None, runner=None
):
    runner = runner or InMemoryRunner(agent=agent, app_name="sec")
    if (
        await runner.session_service.get_session(
            app_name="sec", user_id=user_id, session_id="s"
        )
        is None
    ):
        await runner.session_service.create_session(
            app_name="sec", user_id=user_id, session_id="s"
        )
    message = types.Content(role="user", parts=[types.Part(text="お願い")])
    async for _ in runner.run_async(
        user_id=user_id, session_id="s", new_message=message, state_delta=state_delta
    ):
        pass
    return runner


# --- エスカレーション: 利用者の ID を State から読む ---


async def test_escalation_keyed_by_state_user_id_can_target_another_user() -> None:
    """教材は利用者の ID を State から読む。攻撃者が state_delta で他人の ID を書くと、その人の警告が積み上がる。"""
    warnings: Counter[str] = Counter()
    authenticated: list[str] = []

    def book_escalation(callback_context, llm_request):
        del llm_request
        warnings[callback_context.state.get("user_id", "unknown")] += 1
        authenticated.append(callback_context.user_id)

    agent = Agent(
        name="support",
        model=Calls(model="m"),
        instruction="x",
        tools=[lookup],
        before_model_callback=book_escalation,
    )
    await run(agent, user_id="attacker", state_delta={"user_id": "victim"})
    assert warnings["victim"] > 0
    assert warnings["attacker"] == 0
    # ADK が認証した利用者の ID は、コールバックの context の user_id にある。
    assert set(authenticated) == {"attacker"}


# --- 実行回数の制限: 回数を State に数える ---


def book_limiter(limit: int):
    def check_limit(tool, args, tool_context):
        del tool, args
        total = tool_context.state.get("_total_tool_calls", 0)
        if total >= limit:
            return {"error": f"上限 ({limit}回) に達しました"}
        tool_context.state["_total_tool_calls"] = total + 1
        return None

    return check_limit


async def test_state_counter_can_be_reset_by_the_caller() -> None:
    """上限に達しても、呼び出し元が state_delta で回数を 0 に戻すと、また実行できる。"""
    executed: list[dict] = []

    def record(tool, args, tool_context, tool_response):
        del tool, args, tool_context
        executed.append(tool_response)

    agent = Agent(
        name="support",
        model=Calls(model="m"),
        instruction="x",
        tools=[lookup],
        before_tool_callback=book_limiter(1),
        after_tool_callback=record,
    )
    runner = await run(agent)
    await run(agent, runner=runner)
    await run(agent, runner=runner, state_delta={"_total_tool_calls": 0})
    assert [("error" in r) for r in executed] == [False, True, False]


async def test_state_counter_counts_parallel_calls() -> None:
    """1 ターンで 3 つ並行に呼んでも、回数は 3 増える。後の呼び出しの前のコールバックは、先の書き込みを読める。"""
    agent = Agent(
        name="support",
        model=Calls(model="m", n=3),
        instruction="x",
        tools=[lookup],
        before_tool_callback=book_limiter(100),
    )
    runner = await run(agent)
    session = await runner.session_service.get_session(
        app_name="sec", user_id="u", session_id="s"
    )
    assert session.state["_total_tool_calls"] == 3


# --- 監査ログ: セッション ID を State から読む ---


async def test_session_id_is_not_in_state() -> None:
    """ADK は State にセッション ID を入れない。教材の書き方では常に unknown になる。"""
    seen: list[tuple[str, str]] = []

    def audit(callback_context, llm_request):
        del llm_request
        seen.append(
            (
                callback_context.state.get("session_id", "unknown"),
                callback_context.session.id,
            )
        )

    agent = Agent(
        name="support",
        model=Calls(model="m"),
        instruction="x",
        tools=[lookup],
        before_model_callback=audit,
    )
    await run(agent)
    assert seen[0] == ("unknown", "s")
