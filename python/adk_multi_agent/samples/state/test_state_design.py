"""State の設計パターンを実測で固定する。

プレフィックスの寿命そのものは samples/context/test_state_and_instruction.py が
既に押さえているので、ここでは設計の側だけを見る。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import ValidationError

from samples.state.keys import (
    DEFAULT_RESULTS,
    MAX_RESULTS,
    StateKeys,
    UserState,
    branch_key,
    get_max_results,
    get_user_tier,
    load_user_state,
    looks_sensitive,
    scope_of,
)


class Say(BaseLlm):
    """決まった文を返すモデル。"""

    body: str = "ok"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=self.body)])
        )


def user(body: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=body)])


# --- 鍵の設計 -------------------------------------------------------------


def test_key_constants_carry_their_prefix() -> None:
    """定数を見ただけで寿命が分かる形にする。"""
    assert scope_of(StateKeys.APP_VERSION) == "app"
    assert scope_of(StateKeys.USER_TIER) == "user"
    assert scope_of(StateKeys.TEMP_SEARCH_COUNT) == "temp"


def test_branch_key_separates_parallel_writers() -> None:
    assert branch_key("temp:result", "flight") == "temp:result_flight"
    assert branch_key("temp:result", "hotel") != branch_key("temp:result", "flight")


def test_sensitive_keys_are_rejected() -> None:
    """State は開発 UI にもログにも出る。認証情報は置かない。"""
    assert looks_sensitive("user:auth_token")
    assert looks_sensitive("user:API_KEY")
    assert looks_sensitive("temp:db_password")
    assert not looks_sensitive(StateKeys.USER_TIER)


# --- 型を確かめてから読む -------------------------------------------------


def test_unknown_tier_falls_back_to_free() -> None:
    assert get_user_tier({StateKeys.USER_TIER: "premium"}) == "premium"
    assert get_user_tier({StateKeys.USER_TIER: "vip"}) == "free"
    assert get_user_tier({}) == "free"


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        (10, 10),
        (0, 1),
        (10_000, MAX_RESULTS),
        ("20", DEFAULT_RESULTS),
        (True, DEFAULT_RESULTS),
        (None, DEFAULT_RESULTS),
    ],
)
def test_max_results_is_clamped_and_type_checked(raw: object, want: int) -> None:
    """設定値は人が書くので、型も桁も崩れる前提で読む。

    bool は int の派生なので、明示的に弾かないと True が 1 として通る。
    """
    assert get_max_results({StateKeys.APP_MAX_RESULTS: raw}) == want


def test_load_user_state_does_not_raise_on_bad_values() -> None:
    """読み出しで落とすと対話そのものが止まる。既定値へ倒す。"""
    loaded = load_user_state({StateKeys.USER_TIER: "vip", StateKeys.USER_NAME: "太郎"})
    assert loaded.tier == "free"
    assert loaded.name == "太郎"


def test_user_state_rejects_bad_values_when_constructed_directly() -> None:
    """モデルを直接組むときは弾く。緩めるのは読み出しの層だけにする。"""
    with pytest.raises(ValidationError):
        UserState(tier="vip")


# --- output_key の寿命 ----------------------------------------------------


async def run_pipeline(output_key: str) -> tuple[object, dict]:
    """先行が output_key で書き、後続が同じ Invocation 内で読む。"""
    seen: dict[str, object] = {}

    def peek(callback_context: CallbackContext, llm_request: LlmRequest):
        # 値を覗くだけで、モデル呼び出しは止めない。
        # 暗黙の None がそのまま「処理を続ける」の意味になる。
        seen["value"] = callback_context.state.get(output_key)

    analyzer = Agent(
        name="analyzer",
        model=Say(model="s", body="意図は返品"),
        instruction="分析",
        output_key=output_key,
    )
    responder = Agent(
        name="responder",
        model=Say(model="s", body="返答"),
        instruction="応答",
        before_model_callback=peek,
    )
    runner = InMemoryRunner(
        agent=SequentialAgent(name="pipe", sub_agents=[analyzer, responder]),
        app_name="demo",
    )
    await runner.session_service.create_session(
        app_name="demo", user_id="u", session_id="s1"
    )
    async for _ in runner.run_async(
        user_id="u", session_id="s1", new_message=user("go")
    ):
        pass
    session = await runner.session_service.get_session(
        app_name="demo", user_id="u", session_id="s1"
    )
    return seen.get("value"), session.state


async def test_temp_output_key_reaches_the_next_agent_but_is_not_stored() -> None:
    """中間結果の受け渡しは temp: で足りる。Session には残らない。"""
    seen, state = await run_pipeline("temp:analysis")
    assert seen == "意図は返品"
    assert state == {}


async def test_plain_output_key_is_stored_in_the_session() -> None:
    seen, state = await run_pipeline("analysis")
    assert seen == "意図は返品"
    assert state == {"analysis": "意図は返品"}


# --- 並列の競合 -----------------------------------------------------------


async def run_parallel(key_a: str, key_b: str) -> tuple[dict, list[dict]]:
    a = Agent(
        name="flight",
        model=Say(model="s", body="航空券"),
        instruction="x",
        output_key=key_a,
    )
    b = Agent(
        name="hotel",
        model=Say(model="s", body="ホテル"),
        instruction="x",
        output_key=key_b,
    )
    runner = InMemoryRunner(
        agent=ParallelAgent(name="par", sub_agents=[a, b]), app_name="demo"
    )
    await runner.session_service.create_session(
        app_name="demo", user_id="u", session_id="s1"
    )
    async for _ in runner.run_async(
        user_id="u", session_id="s1", new_message=user("go")
    ):
        pass
    session = await runner.session_service.get_session(
        app_name="demo", user_id="u", session_id="s1"
    )
    deltas = [
        e.actions.state_delta
        for e in session.events
        if e.actions and e.actions.state_delta
    ]
    return session.state, deltas


async def test_same_key_in_parallel_loses_a_value_without_any_error() -> None:
    """後勝ちで先の値が消える。例外も警告も出ないので、組み立て時に落とす。

    イベントの state_delta には両方残るため、後から追うことはできる。
    """
    state, deltas = await run_parallel("result", "result")
    assert state == {"result": "ホテル"}
    assert {"result": "航空券"} in deltas
    assert {"result": "ホテル"} in deltas


async def test_separate_keys_keep_both_results() -> None:
    state, _ = await run_parallel(
        branch_key("result", "flight"), branch_key("result", "hotel")
    )
    assert state == {"result_flight": "航空券", "result_hotel": "ホテル"}
