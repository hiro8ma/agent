"""ADK の A2A のサーバー（to_a2a）とクライアント（RemoteA2aAgent）を確かめる。モデルは呼ばない。"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import httpx
import pytest
from google.adk import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import request_input
from google.genai import types

GO_DIR = Path(__file__).resolve().parents[4] / "go"


class Ask(BaseLlm):
    """「不足」を含む発話には request_input で聞き返し、それ以外は受け付ける。"""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        text = llm_request.contents[-1].parts[0].text or ""
        if "不足" in text:
            call = types.FunctionCall(
                name="adk_request_input", args={"message": "金額を教えてください"}
            )
            part = types.Part(function_call=call)
        else:
            part = types.Part(text="受け付けました")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def v0_send(text: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "messageId": f"m-{text}",
                "role": "user",
                "kind": "message",
                "parts": [{"kind": "text", "text": text}],
            }
        },
    }


@pytest.fixture
async def python_server() -> AsyncGenerator[httpx.AsyncClient, None]:
    agent = Agent(
        name="expense_agent",
        model=Ask(model="gemini-3.8-flash"),
        instruction="x",
        tools=[request_input],
    )
    app = to_a2a(agent, host="0.0.0.0", port=8001)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client


@pytest.mark.parametrize(
    ("text", "state"),
    [("交通費1280円", "completed"), ("情報が不足しています", "input-required")],
    ids=["そろっていれば完了", "request_input で聞き返すと入力待ち"],
)
async def test_request_input_becomes_input_required(
    python_server: httpx.AsyncClient, text: str, state: str
) -> None:
    res = (await python_server.post("/", json=v0_send(text))).json()
    assert res["result"]["status"]["state"] == state


async def test_auto_card_advertises_the_bind_address(
    python_server: httpx.AsyncClient,
) -> None:
    """Agent Card を省くと、待ち受けの 0.0.0.0 がそのまま url に入り、ほかのマシンからは使えない。"""
    card = (await python_server.get("/.well-known/agent-card.json")).json()
    assert card["url"] == "http://0.0.0.0:8001"
    assert card["protocolVersion"].startswith("0.3")


async def test_python_server_rejects_v1_requests(
    python_server: httpx.AsyncClient,
) -> None:
    v1 = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": "m",
                "role": "ROLE_USER",
                "parts": [{"text": "hi"}],
            }
        },
    }
    res = (await python_server.post("/", json=v1)).json()
    assert res["error"]["code"] == -32601


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def go_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Go の ADK の A2A v1.0 のサーバー（v0.3 の互換の口つき）を起動する。"""
    if shutil.which("go") is None:
        pytest.skip("go が無い")
    binary = tmp_path_factory.mktemp("go") / "a2a-interop-server"
    subprocess.run(
        ["go", "build", "-o", str(binary), "./cmd/a2a-interop-server"],
        cwd=GO_DIR,
        check=True,
        capture_output=True,
    )
    addr = f"127.0.0.1:{free_port()}"
    proc = subprocess.Popen([str(binary), "-addr", addr])
    try:
        for _ in range(50):
            try:
                httpx.get(f"http://{addr}/.well-known/agent-card.json", timeout=0.2)
                break
            except httpx.TransportError:
                time.sleep(0.1)
        yield f"http://{addr}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


async def test_python_remote_agent_calls_go_server(go_server: str) -> None:
    """Python の ADK（a2a-sdk 0.3）の RemoteA2aAgent から、Go の ADK の v1.0 のサーバーを互換の口経由で呼べる。"""
    remote = RemoteA2aAgent(
        name="go_expense", agent_card=f"{go_server}/.well-known/agent-card.json"
    )
    runner = InMemoryRunner(agent=remote, app_name="interop")
    await runner.session_service.create_session(
        app_name="interop", user_id="u", session_id="s"
    )
    texts = []
    message = types.Content(role="user", parts=[types.Part(text="交通費1280円")])
    async for ev in runner.run_async(user_id="u", session_id="s", new_message=message):
        texts += [p.text for p in (ev.content.parts if ev.content else []) if p.text]
    assert any("Go で受け付けました: 交通費1280円" in t for t in texts), texts
