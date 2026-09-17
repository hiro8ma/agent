"""3 層の寿命と、保存の経路を実測で固定する。

教材の説明と実物が食い違った箇所を、落ちる検査として置く。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions
from google.adk.memory import InMemoryMemoryService
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from samples.memory.agent import (
    build_agent,
    is_persisted,
    scope_of,
    visible_across_sessions,
)


class ScriptedModel(BaseLlm):
    """台本の順に応答を返すモデル。"""

    turns: list[types.Content] = []
    calls: int = 0
    seen: list[str] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.seen.append(llm_request.config.system_instruction or "")
        i = self.calls
        self.calls = i + 1
        if i >= len(self.turns):
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="了解")])
            )
            return
        yield LlmResponse(content=self.turns[i])


def text(body: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=body)])


def user(body: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=body)])


def call(name: str, args: dict) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
    )


# --- State のスコープ判定 -------------------------------------------------


def test_scope_is_decided_by_the_prefix() -> None:
    assert scope_of("app:version") == "app"
    assert scope_of("user:tier") == "user"
    assert scope_of("temp:scratch") == "temp"
    assert scope_of("draft") == "session"


def test_only_temp_is_dropped_before_storage() -> None:
    assert is_persisted("app:version")
    assert is_persisted("user:tier")
    assert is_persisted("draft")
    assert not is_persisted("temp:scratch")


def test_session_scope_does_not_cross_sessions() -> None:
    assert visible_across_sessions("app:version")
    assert visible_across_sessions("user:tier")
    assert not visible_across_sessions("draft")
    assert not visible_across_sessions("temp:scratch")


# --- 保存の経路 -----------------------------------------------------------


async def test_assigning_to_session_state_does_not_persist() -> None:
    """教材の `session.state[key] = value` は保存されない。

    Session.state は素の dict で、get_session が返すのはコピーになる。
    """
    service = InMemorySessionService()
    session = await service.create_session(
        app_name="demo", user_id="u1", state={"user:name": "初期値"}
    )
    assert isinstance(session.state, dict)

    session.state["user:name"] = "書き換えた"

    fetched = await service.get_session(
        app_name="demo", user_id="u1", session_id=session.id
    )
    assert fetched.state["user:name"] == "初期値"
    assert fetched is not session


async def test_state_delta_on_an_event_persists() -> None:
    """保存されるのは、イベントの state_delta に乗った値だけ。"""
    service = InMemorySessionService()
    session = await service.create_session(app_name="demo", user_id="u1")

    await service.append_event(
        session=session,
        event=Event(
            author="user",
            actions=EventActions(
                state_delta={"user:name": "イベント経由", "temp:t": "消える"}
            ),
            content=user("hi"),
        ),
    )

    fetched = await service.get_session(
        app_name="demo", user_id="u1", session_id=session.id
    )
    assert fetched.state["user:name"] == "イベント経由"
    assert "temp:t" not in fetched.state


async def test_tool_context_state_is_the_write_path() -> None:
    """ツールの中の state は State 型で、書いた値がイベントに乗る。"""
    seen: list[str] = []

    def remember(note: str, tool_context: ToolContext) -> dict:
        """メモを残す。"""
        seen.append(type(tool_context.state).__name__)
        tool_context.state["user:note"] = note
        tool_context.state["temp:scratch"] = "消える"
        tool_context.state["draft"] = "残る"
        return {"saved": note}

    model = ScriptedModel(
        model="scripted",
        turns=[call("remember", {"note": "うどん"}), text("覚えました")],
    )
    agent = Agent(name="memo", model=model, instruction="memo", tools=[remember])
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    async for _ in runner.run_async(
        user_id="u1", session_id="s1", new_message=user("覚えて")
    ):
        pass

    assert seen == ["State"]
    session = await runner.session_service.get_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    assert session.state["user:note"] == "うどん"
    assert session.state["draft"] == "残る"
    assert "temp:scratch" not in session.state

    deltas = [e.actions.state_delta for e in session.events if e.actions]
    assert {"user:note": "うどん", "draft": "残る"} in deltas
    assert all("temp:scratch" not in d for d in deltas)


async def test_user_scope_reaches_a_new_session_but_not_another_user() -> None:
    service = InMemorySessionService()
    session = await service.create_session(app_name="demo", user_id="u1")
    await service.append_event(
        session=session,
        event=Event(
            author="user",
            actions=EventActions(
                state_delta={"user:tier": "premium", "app:version": "2.1.0"}
            ),
            content=user("hi"),
        ),
    )

    same_user = await service.create_session(app_name="demo", user_id="u1")
    assert same_user.state["user:tier"] == "premium"
    assert same_user.state["app:version"] == "2.1.0"

    other_user = await service.create_session(app_name="demo", user_id="u2")
    assert "user:tier" not in other_user.state
    assert other_user.state["app:version"] == "2.1.0"


# --- Memory ---------------------------------------------------------------


async def test_memory_is_scoped_to_app_and_user() -> None:
    """Memory は利用者もアプリも横断しない。鍵は (app_name, user_id) になる。"""
    sessions = InMemorySessionService()
    memories = InMemoryMemoryService()
    session = await sessions.create_session(app_name="demo", user_id="u1")
    await sessions.append_event(
        session=session, event=Event(author="user", content=user("I like udon"))
    )
    session = await sessions.get_session(
        app_name="demo", user_id="u1", session_id=session.id
    )
    await memories.add_session_to_memory(session)

    hit = await memories.search_memory(app_name="demo", user_id="u1", query="udon")
    assert len(hit.memories) == 1

    other_user = await memories.search_memory(
        app_name="demo", user_id="u2", query="udon"
    )
    assert other_user.memories == []

    other_app = await memories.search_memory(
        app_name="other", user_id="u1", query="udon"
    )
    assert other_app.memories == []


async def test_in_memory_search_never_matches_japanese() -> None:
    """InMemoryMemoryService の語の切り出しは [A-Za-z]+ なので日本語は当たらない。

    docstring にも prototyping purpose only と書いてある。
    日本語で試して「Memory が動かない」と誤解しないための検査。
    """
    sessions = InMemorySessionService()
    memories = InMemoryMemoryService()
    session = await sessions.create_session(app_name="demo", user_id="u1")
    await sessions.append_event(
        session=session,
        event=Event(author="user", content=user("私はうどんが好きです")),
    )
    session = await sessions.get_session(
        app_name="demo", user_id="u1", session_id=session.id
    )
    await memories.add_session_to_memory(session)

    hit = await memories.search_memory(app_name="demo", user_id="u1", query="うどん")
    assert hit.memories == []


async def test_after_agent_callback_saves_the_whole_turn() -> None:
    """保存の時点で、その回の最終応答までイベントに入っている。"""
    counted: list[int] = []

    async def save(callback_context: CallbackContext) -> None:
        counted.append(len(callback_context.get_invocation_context().session.events))
        await callback_context.add_session_to_memory()

    model = ScriptedModel(model="scripted", turns=[text("覚えました")])
    agent = build_agent(model)
    agent.after_agent_callback = save
    runner = InMemoryRunner(agent=agent, app_name="demo")
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    async for _ in runner.run_async(
        user_id="u1", session_id="s1", new_message=user("I like udon")
    ):
        pass

    assert counted == [2]
    hit = await runner.memory_service.search_memory(
        app_name="demo", user_id="u1", query="udon"
    )
    assert [m.author for m in hit.memories] == ["user"]


async def test_preload_memory_injects_past_conversations() -> None:
    """次の Session では、検索結果が instruction へ足されて届く。"""
    runner = InMemoryRunner(
        agent=build_agent(ScriptedModel(model="scripted", turns=[text("覚えました")])),
        app_name="demo",
    )
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    async for _ in runner.run_async(
        user_id="u1", session_id="s1", new_message=user("I like udon")
    ):
        pass

    second = ScriptedModel(model="scripted", turns=[text("うどんですね")])
    runner2 = Runner(
        app_name="demo",
        agent=build_agent(second),
        session_service=runner.session_service,
        memory_service=runner.memory_service,
    )
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s2"
    )
    async for _ in runner2.run_async(
        user_id="u1", session_id="s2", new_message=user("what udon do I like")
    ):
        pass

    assert "<PAST_CONVERSATIONS>" in second.seen[-1]
    assert "I like udon" in second.seen[-1]


async def test_missing_memory_service_fails_loudly() -> None:
    """memory_service を渡さないと、保存の時点で実行が落ちる。"""
    runner = Runner(
        app_name="demo",
        agent=build_agent(ScriptedModel(model="scripted", turns=[text("はい")])),
        session_service=InMemorySessionService(),
    )
    await runner.session_service.create_session(
        app_name="demo", user_id="u1", session_id="s1"
    )
    with pytest.raises(ValueError, match="memory service is not available"):
        async for _ in runner.run_async(
            user_id="u1", session_id="s1", new_message=user("hi")
        ):
            pass
