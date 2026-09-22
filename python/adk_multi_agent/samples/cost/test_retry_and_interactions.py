"""Gemini の再試行の既定値と、Interactions API の 404 のときの ADK の振る舞いを、偽のサーバーで確かめる。"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from google import genai
from google.adk import Agent
from google.adk.events import Event
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import errors, types
from google.genai._gaos.lib import compat_errors


@dataclass
class Fake:
    """全ての要求に同じステータスを返し、要求の時刻と本文を記録する。"""

    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    requests: list[tuple[float, str, dict]] = field(default_factory=list)
    url: str = ""


@pytest.fixture
def fake() -> Iterator[Fake]:
    state = Fake()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            state.requests.append((time.monotonic(), self.path, body))
            code = {
                400: "INVALID_ARGUMENT",
                404: "NOT_FOUND",
                429: "RESOURCE_EXHAUSTED",
            }
            if self.path.endswith("/interactions"):
                # Interactions API のエラーの code は数値ではなく文字列。
                payload = {"error": {"code": "fake-error", "message": "fake"}}
            else:
                payload = {
                    "error": {
                        "code": state.status,
                        "message": "fake",
                        "status": code.get(state.status, "INTERNAL"),
                    }
                }
            raw = json.dumps(payload).encode()
            self.send_response(state.status)
            self.send_header("content-type", "application/json")
            for k, v in state.headers.items():
                self.send_header(k, v)
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()


def call(fake: Fake, retry: types.HttpRetryOptions | None) -> None:
    client = genai.Client(
        api_key="fake-key",
        http_options=types.HttpOptions(base_url=fake.url, retry_options=retry),
    )
    client.models.generate_content(model="gemini-3.8-flash", contents="こんにちは")


FAST = types.HttpRetryOptions(initial_delay=0.05, max_delay=0.2, jitter=0)


@pytest.mark.parametrize(
    ("status", "retry", "attempts"),
    [
        (429, None, 1),
        (503, None, 1),
        (429, FAST, 5),
        (500, FAST, 5),
        (503, FAST, 5),
        (504, FAST, 5),
        (400, FAST, 1),
        (404, FAST, 1),
    ],
    ids=[
        "再試行の設定が無ければ429でも1回",
        "再試行の設定が無ければ503でも1回",
        "設定すると429は5回",
        "設定すると500も5回",
        "設定すると503は5回",
        "設定すると504も5回",
        "400は再試行しない",
        "404は再試行しない",
    ],
)
def test_retry_defaults(fake: Fake, status: int, retry, attempts: int) -> None:
    fake.status = status
    with pytest.raises(errors.APIError) as caught:
        call(fake, retry)
    assert caught.value.code == status
    assert len(fake.requests) == attempts


def test_retry_ignores_retry_after(fake: Fake) -> None:
    """429 の Retry-After を読まず、指数バックオフの間隔で打ち直す。"""
    fake.status = 429
    fake.headers = {"retry-after": "5"}
    with pytest.raises(errors.APIError):
        call(fake, types.HttpRetryOptions(attempts=2, initial_delay=0.05, jitter=0))
    (t1, _, _), (t2, _, _) = fake.requests
    assert t2 - t1 < 1


def test_adk_gemini_does_not_retry_by_default() -> None:
    """ADK の Agent(model="gemini-...") は retry_options を持たず、再試行しない。"""
    agent = Agent(name="support", model="gemini-3.8-flash", instruction="x")
    assert agent.canonical_model.retry_options is None


async def test_stale_interaction_id_is_resent(
    fake: Fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactions API で前回の interaction が消えて 404 になっても、次の呼び出しも同じ ID を送る。"""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    fake.status = 404
    model = Gemini(
        model="gemini-3.8-flash", use_interactions_api=True, base_url=fake.url
    )
    agent = Agent(name="support", model=model, instruction="x")
    runner = InMemoryRunner(agent=agent, app_name="app")
    session = await runner.session_service.create_session(
        app_name="app", user_id="u", session_id="s"
    )
    await runner.session_service.append_event(
        session,
        Event(
            author="support",
            interaction_id="stale-1",
            content=types.Content(role="model", parts=[types.Part(text="前回の答え")]),
        ),
    )

    for text in ["1回目", "2回目"]:
        message = types.Content(role="user", parts=[types.Part(text=text)])
        with pytest.raises(compat_errors.NotFoundError) as caught:
            async for _ in runner.run_async(
                user_id="u", session_id="s", new_message=message
            ):
                pass
        # generate_content の経路とは例外の型が違い、genai.errors.APIError では捕まらない。
        assert not isinstance(caught.value, errors.APIError)

    sent = [body.get("previous_interaction_id") for _, _, body in fake.requests]
    assert sent == ["stale-1", "stale-1"]
    assert all(path.endswith("/interactions") for _, path, _ in fake.requests)


@pytest.mark.parametrize(
    ("status", "attempts"),
    [(429, 4), (500, 4), (503, 4), (400, 1), (404, 1)],
    ids=["429は4回", "500は4回", "503は4回", "400は1回", "404は1回"],
)
async def test_interactions_retries_without_retry_options(
    fake: Fake, status: int, attempts: int
) -> None:
    """Interactions API の経路は、再試行の設定が無くても既定で 3 回打ち直す（計 4 回）。"""
    fake.status = status
    client = genai.Client(
        api_key="fake-key", http_options=types.HttpOptions(base_url=fake.url)
    )
    with pytest.raises(compat_errors.APIError):
        await client.aio.interactions.create(
            model="gemini-3.8-flash", input="こんにちは"
        )
    assert len(fake.requests) == attempts
