"""SequentialAgent にリモートの A2A エージェントを並べたとき、各段のサーバーに何が届くかを確かめる。"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator

import httpx
from google.adk import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import SequentialAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types


class Stage(BaseLlm):
    """受け取った利用者の発話を記録し、決まった文を返す。"""

    tag: str = ""
    seen: list[str] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        user = [c for c in llm_request.contents if c.role == "user"]
        self.seen = [*self.seen, "".join(p.text or "" for p in user[-1].parts or [])]
        yield LlmResponse(
            content=types.Content(
                role="model", parts=[types.Part(text=f"{self.tag}の結果")]
            )
        )


class Router(httpx.AsyncBaseTransport):
    """ホスト名ごとに、メモリ内の ASGI アプリへ振り分ける。"""

    def __init__(self, apps: dict) -> None:
        self.transports = {
            host: httpx.ASGITransport(app=app) for host, app in apps.items()
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self.transports[request.url.host].handle_async_request(request)


async def test_each_stage_receives_all_previous_outputs() -> None:
    """前の段の出力は次の段の入力に置き換わらず、元の依頼の後ろにすべて連結されて届く。"""
    names = ["collector", "analyzer", "reporter"]
    models = {n: Stage(model="gemini-3.8-flash", tag=n) for n in names}
    apps = {
        n: to_a2a(Agent(name=n, model=models[n], instruction="x"), host=n, port=80)
        for n in names
    }
    async with contextlib.AsyncExitStack() as stack:
        for app in apps.values():
            await stack.enter_async_context(app.router.lifespan_context(app))
        client = await stack.enter_async_context(
            httpx.AsyncClient(transport=Router(apps))
        )
        stages = [
            RemoteA2aAgent(
                name=n,
                agent_card=f"http://{n}:80/.well-known/agent-card.json",
                httpx_client=client,
            )
            for n in names
        ]
        runner = InMemoryRunner(
            agent=SequentialAgent(name="pipeline", sub_agents=stages), app_name="p"
        )
        await runner.session_service.create_session(
            app_name="p", user_id="u", session_id="s"
        )
        message = types.Content(
            role="user", parts=[types.Part(text="先月の売上をまとめて")]
        )
        async for _ in runner.run_async(
            user_id="u", session_id="s", new_message=message
        ):
            pass

    assert models["collector"].seen == ["先月の売上をまとめて"]
    assert models["analyzer"].seen == [
        "先月の売上をまとめてFor context:[collector] said: collectorの結果"
    ]
    assert models["reporter"].seen == [
        (
            "先月の売上をまとめて"
            "For context:[collector] said: collectorの結果"
            "For context:[analyzer] said: analyzerの結果"
        )
    ]
