"""A2A の Push Notification の送り手と受け手を確かめる。モデルは台本で置き換える。"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory, create_text_message_object
from a2a.client.errors import A2AClientJSONRPCError
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    PushNotificationAuthenticationInfo,
    PushNotificationConfig,
    TaskPushNotificationConfig,
)
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from google.adk import Agent
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

BASE = "http://localhost:8001"
WEBHOOK = "http://webhook.local/webhook/a2a-notification"
SECRET = "webhook-secret"


def card(push: bool) -> AgentCard:
    return AgentCard(
        name="expense",
        description="経費",
        url=BASE,
        version="1.0.0",
        capabilities=AgentCapabilities(push_notifications=push),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )


class Ok(BaseLlm):
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="ok")])
        )


def agent() -> Agent:
    return Agent(name="expense", model=Ok(model="gemini-3.8-flash"), instruction="x")


def book_webhook(received: list[dict]) -> FastAPI:
    """教材の Webhook。受け取った要求を記録する。"""
    app = FastAPI()

    @app.post("/webhook/a2a-notification")
    async def handle_notification(request: Request):
        body = await request.json()
        received.append(
            {
                "authorization": request.headers.get("authorization"),
                "token": request.headers.get("x-a2a-notification-token"),
                "state": body.get("status", {}).get("state"),
            }
        )
        if request.headers.get("Authorization", "") != f"Bearer {SECRET}":
            return {"error": "Unauthorized"}, 401
        return {"status": "received"}

    return app


def push_config() -> PushNotificationConfig:
    """教材の設定に token も足す。"""
    return PushNotificationConfig(
        url=WEBHOOK,
        token="per-task-token",
        authentication=PushNotificationAuthenticationInfo(
            schemes=["Bearer"], credentials=SECRET
        ),
    )


def wired_app(webhook: FastAPI):
    """to_a2a を使わず、送り手を組み込んだ A2A のサーバー。"""
    store = InMemoryPushNotificationConfigStore()
    sender_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=webhook))
    handler = DefaultRequestHandler(
        agent_executor=A2aAgentExecutor(
            runner=InMemoryRunner(agent=agent(), app_name="expense")
        ),
        task_store=InMemoryTaskStore(),
        push_config_store=store,
        push_sender=BasePushNotificationSender(sender_client, store),
    )
    return A2AStarletteApplication(agent_card=card(True), http_handler=handler).build()


async def send(app, configs: list[PushNotificationConfig], lifespan: bool) -> str:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE
    ) as hc:
        client_config = ClientConfig(
            httpx_client=hc, streaming=False, push_notification_configs=configs
        )
        if lifespan:
            async with app.router.lifespan_context(app):
                return await _send(client_config)
        return await _send(client_config)


async def _send(client_config: ClientConfig) -> str:
    client = await ClientFactory.connect(BASE, client_config=client_config)
    async for event in client.send_message(create_text_message_object(content="経費")):
        if isinstance(event, tuple):
            return event[0].status.state.value
    return "none"


async def test_to_a2a_accepts_config_but_never_sends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """to_a2a は設定の保存先を作るが送り手を組み込まないので、Webhook への送信が 1 件も起きない。"""
    original = httpx.AsyncClient.send
    webhook_calls: list[str] = []

    async def recording_send(self, request, **kwargs):
        if request.url.host == "webhook.local":
            webhook_calls.append(str(request.url))
            return httpx.Response(200, json={}, request=request)
        return await original(self, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", recording_send)
    app = to_a2a(agent(), host="localhost", port=8001, agent_card=card(True))
    assert await send(app, [push_config()], lifespan=True) == "completed"
    assert webhook_calls == []

    wired = wired_app(book_webhook([]))
    assert await send(wired, [push_config()], lifespan=False) == "completed"
    assert webhook_calls, "送り手を組み込めば、同じ記録の仕方で送信が見える"


async def test_sender_uses_token_header_not_credentials() -> None:
    """a2a-sdk 0.3 の送り手は token を X-A2A-Notification-Token で送り、authentication の credentials は送らない。"""
    received: list[dict] = []
    app = wired_app(book_webhook(received))
    assert await send(app, [push_config()], lifespan=False) == "completed"
    assert received
    assert {r["authorization"] for r in received} == {None}
    assert {r["token"] for r in received} == {"per-task-token"}
    assert received[-1]["state"] == "completed"


def test_book_webhook_rejects_with_status_200() -> None:
    """教材の Webhook は、拒否のつもりのタプルを JSON の配列にし、200 で返す。"""
    client = TestClient(book_webhook([]))
    res = client.post(
        "/webhook/a2a-notification", json={"status": {"state": "completed"}}
    )
    assert res.status_code == 200
    assert res.json() == [{"error": "Unauthorized"}, 401]


async def test_set_callback_requires_push_capability() -> None:
    """Agent Card で push_notifications を宣言しないと、set_task_callback は拒まれる。"""
    app = to_a2a(agent(), host="localhost", port=8001, agent_card=card(False))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as hc,
    ):
        client = await ClientFactory.connect(
            BASE, client_config=ClientConfig(httpx_client=hc, streaming=False)
        )
        task_id = None
        async for event in client.send_message(
            create_text_message_object(content="経費")
        ):
            if isinstance(event, tuple):
                task_id = event[0].id
        assert task_id is not None
        config = TaskPushNotificationConfig(
            task_id=task_id, push_notification_config=push_config()
        )
        with pytest.raises(A2AClientJSONRPCError) as caught:
            await client.set_task_callback(config)
        assert "not supported" in str(caught.value)
