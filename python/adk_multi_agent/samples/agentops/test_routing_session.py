"""同じセッションで続けて問い合わせたとき、どのエージェントが答えるかを AdkApp で確かめる。モデルは台本で置き換える。"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk import Agent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types
from vertexai.agent_engines import AdkApp

SCENARIOS = [
    "製品Aの返品方法を教えてください",
    "製品Aの技術仕様を他社製品と比較してください",
    "こんにちは",
]


def reply(part: types.Part) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[part]))


class Root(BaseLlm):
    """返品は FAQ、比較は調査へ振り分け、それ以外は自分で答える。"""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        text = llm_request.contents[-1].parts[-1].text or ""
        to = (
            "faq_agent"
            if "返品" in text
            else "research_agent"
            if "比較" in text
            else None
        )
        if to:
            call = types.FunctionCall(name="transfer_to_agent", args={"agent_name": to})
            yield reply(types.Part(function_call=call))
        else:
            yield reply(types.Part(text="ご用件をどうぞ"))


class Leaf(BaseLlm):
    """振り分けはせず、担当の範囲で答える。"""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        yield reply(types.Part(text="回答"))


def tree(disallow_transfer_to_parent: bool) -> Agent:
    def leaf(name: str) -> Agent:
        return Agent(
            name=name,
            model=Leaf(model=name),
            instruction="x",
            description=name,
            disallow_transfer_to_parent=disallow_transfer_to_parent,
        )

    return Agent(
        name="support_agent",
        model=Root(model="root"),
        instruction="x",
        sub_agents=[leaf("faq_agent"), leaf("research_agent")],
    )


async def answered_by(app: AdkApp, session_id: str, message: str) -> str:
    final = None
    async for event in app.async_stream_query(
        user_id="test-user", session_id=session_id, message=message
    ):
        final = event
    return final["author"]


@pytest.mark.parametrize(
    ("disallow", "want"),
    [
        (False, ["faq_agent", "faq_agent", "faq_agent"]),
        (True, ["faq_agent", "research_agent", "support_agent"]),
    ],
    ids=[
        "既定では2回目以降も最初に振り分けた先が答える",
        "親への移譲を禁じると毎ターンがルートから始まる",
    ],
)
async def test_same_session_routing(disallow: bool, want: list[str]) -> None:
    app = AdkApp(agent=tree(disallow))
    session = await app.async_create_session(user_id="test-user")
    got = [await answered_by(app, session["id"], m) for m in SCENARIOS]
    assert got == want


async def test_separate_sessions_route_from_root() -> None:
    """シナリオごとにセッションを分ければ、既定の構成でも毎回ルートが振り分ける。"""
    app = AdkApp(agent=tree(False))
    got = []
    for message in SCENARIOS:
        session = await app.async_create_session(user_id="test-user")
        got.append(await answered_by(app, session["id"], message))
    assert got == ["faq_agent", "research_agent", "support_agent"]
