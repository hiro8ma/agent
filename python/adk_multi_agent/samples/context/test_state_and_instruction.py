"""State の保存範囲と三種類の Instruction を実物で確かめる。"""

from __future__ import annotations

from types import SimpleNamespace

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.models import LlmRequest
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.sessions import InMemorySessionService

from samples.context.instructions import (
    GLOBAL_POLICY,
    app,
    order_agent,
    root_agent,
    support_agent,
    support_instruction,
)


def _readonly(state: dict) -> ReadonlyContext:
    invocation = SimpleNamespace(
        session=SimpleNamespace(state=state),
        user_content=None,
        invocation_id="invocation-1",
        agent=support_agent,
        user_id="user-1",
        run_config=None,
        credential_by_key={},
    )
    return ReadonlyContext(invocation)


async def test_state_prefixes_define_lifetime_and_audience():
    service = InMemorySessionService()
    first = await service.create_session(
        app_name="shop",
        user_id="user-1",
        session_id="session-1",
        state={
            "app:policy_version": "v1",
            "user:display_name": "田中",
            "issue_kind": "返品",
            "temp:draft": "一時回答",
        },
    )
    assert first.state == {
        "app:policy_version": "v1",
        "user:display_name": "田中",
        "issue_kind": "返品",
    }

    same_user = await service.create_session(
        app_name="shop", user_id="user-1", session_id="session-2"
    )
    assert same_user.state == {
        "app:policy_version": "v1",
        "user:display_name": "田中",
    }

    other_user = await service.create_session(
        app_name="shop", user_id="user-2", session_id="session-3"
    )
    assert other_user.state == {"app:policy_version": "v1"}

    other_app = await service.create_session(
        app_name="other", user_id="user-1", session_id="session-4"
    )
    assert other_app.state == {}


async def test_temp_state_is_visible_in_invocation_but_not_persisted():
    service = InMemorySessionService()
    session = await service.create_session(
        app_name="shop", user_id="user-1", session_id="session-1"
    )

    await service.append_event(
        session,
        Event(
            invocation_id="invocation-1",
            author="order_agent",
            actions=EventActions(state_delta={"temp:order_result": "発送済み"}),
        ),
    )
    assert session.state["temp:order_result"] == "発送済み"

    stored = await service.get_session(
        app_name="shop", user_id="user-1", session_id="session-1"
    )
    assert "temp:order_result" not in stored.state


def test_dynamic_instruction_reads_user_and_session_state():
    instruction = support_instruction(
        _readonly({"user:display_name": "田中", "issue_kind": "接続障害"})
    )
    assert "田中" in instruction
    assert "接続障害" in instruction


async def test_app_registers_global_instruction_for_every_agent():
    assert app.root_agent is root_agent
    assert [agent.name for agent in root_agent.sub_agents] == [
        order_agent.name,
        support_agent.name,
    ]
    assert root_agent.global_instruction == ""

    plugin = app.plugins[0]
    assert isinstance(plugin, GlobalInstructionPlugin)
    request = LlmRequest()
    await plugin.before_model_callback(
        callback_context=_readonly({}),
        llm_request=request,
    )
    assert request.config.system_instruction == GLOBAL_POLICY
