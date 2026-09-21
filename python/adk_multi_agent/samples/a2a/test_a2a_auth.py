"""A2A の認証の宣言と、クライアントとサーバーでの扱いを確かめる。モデルは台本で置き換える。"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import time
from collections.abc import AsyncGenerator, Iterator

import httpx
import pytest
from a2a.client import (
    AuthInterceptor,
    ClientCallContext,
    ClientConfig,
    ClientFactory,
    CredentialService,
    InMemoryContextCredentialStore,
    create_text_message_object,
)
from a2a.client.errors import A2AClientHTTPError
from a2a.types import AgentCard
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from google.adk import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.auth import crypt, jwt
from google.auth.exceptions import GoogleAuthError
from google.genai import types
from google.oauth2 import id_token
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

BASE = "http://localhost:8001"
TOKEN = "test-token"

BOOK_CARD = {
    "name": "secure-expense-agent",
    "description": "経費",
    "url": BASE,
    "version": "1.0.0",
    "capabilities": {},
    "default_input_modes": ["text"],
    "default_output_modes": ["text"],
    "skills": [],
    "security_schemes": {
        "oauth2": {
            "type": "oauth2",
            "flows": {
                "clientCredentials": {
                    "tokenUrl": "https://auth.example.com/oauth/token",
                    "scopes": {
                        "expense:read": "経費の読み取り",
                        "expense:write": "経費の書き込み",
                    },
                }
            },
        }
    },
}

REQUIRED = BOOK_CARD | {"security": [{"oauth2": ["expense:read"]}]}


class Ok(BaseLlm):
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="ok")])
        )


class Fixed(CredentialService):
    async def get_credentials(
        self, security_scheme_name: str, context: ClientCallContext | None
    ) -> str | None:
        del security_scheme_name, context
        return TOKEN


class RequireBearer(BaseHTTPMiddleware):
    """Agent Card の取得だけを認証なしで通し、ほかは Bearer を求める。"""

    async def dispatch(self, request, call_next):
        if request.url.path.startswith("/.well-known/"):
            return await call_next(request)
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def build(card: AgentCard, enforce: bool = False):
    app = to_a2a(
        Agent(name="expense", model=Ok(model="gemini-3.8-flash"), instruction="x"),
        host="localhost",
        port=8001,
        agent_card=card,
    )
    seen: list[str | None] = []

    class Record(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.method == "POST":
                seen.append(request.headers.get("authorization"))
            return await call_next(request)

    if enforce:
        app.add_middleware(RequireBearer)
    app.add_middleware(Record)
    return app, seen


async def send(app, interceptors=None, context=None) -> str:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as hc,
    ):
        client = await ClientFactory.connect(
            BASE,
            client_config=ClientConfig(httpx_client=hc, streaming=False),
            interceptors=interceptors,
        )
        async for event in client.send_message(
            create_text_message_object(content="7月の経費"), context=context
        ):
            task = event[0] if isinstance(event, tuple) else None
            if task is not None:
                return task.status.state.value
    return "none"


def test_book_card_parses_but_has_no_requirement() -> None:
    """教材の snake_case の JSON も読めるが、security が無いので要件は宣言されない。"""
    card = AgentCard.model_validate(BOOK_CARD)
    assert "oauth2" in card.security_schemes
    assert card.security is None
    dumped = card.model_dump(mode="json", exclude_none=True)
    assert "securitySchemes" in dumped
    assert "security_schemes" not in dumped


async def test_declaring_a_scheme_does_not_enforce_it() -> None:
    """Agent Card で OAuth 2 を宣言しても、サーバーはトークンの無い呼び出しを通す。"""
    card = AgentCard.model_validate(REQUIRED)
    app, seen = build(card)
    assert await send(app) == "completed"
    assert seen == [None]


async def test_interceptor_ignores_schemes_without_security() -> None:
    """security が無い Agent Card では、AuthInterceptor は認証情報を付けない。"""
    app, seen = build(AgentCard.model_validate(BOOK_CARD))
    assert await send(app, [AuthInterceptor(Fixed())]) == "completed"
    assert seen == [None]


async def test_interceptor_attaches_bearer_with_security() -> None:
    """security で要件を書くと、OAuth 2 のスキームに Bearer を付ける。"""
    card = AgentCard.model_validate(REQUIRED)
    app, seen = build(card)
    assert await send(app, [AuthInterceptor(Fixed())]) == "completed"
    assert seen == [f"Bearer {TOKEN}"]


async def test_middleware_enforces_and_keeps_card_public() -> None:
    """強制はサーバーの前段で行う。Agent Card の取得は認証なしで通す。"""
    card = AgentCard.model_validate(REQUIRED)
    app, _ = build(card, enforce=True)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as hc,
    ):
        assert (await hc.get("/.well-known/agent-card.json")).status_code == 200
        assert (
            await hc.post(
                "/", json={"jsonrpc": "2.0", "id": 1, "method": "message/send"}
            )
        ).status_code == 401

    app, seen = build(card, enforce=True)
    assert await send(app, [AuthInterceptor(Fixed())]) == "completed"
    assert seen == [f"Bearer {TOKEN}"]


async def book_client_store(scheme_name: str) -> InMemoryContextCredentialStore:
    store = InMemoryContextCredentialStore()
    await store.set_credentials("default", scheme_name, TOKEN)
    return store


async def test_book_client_default_scheme_name_sends_nothing() -> None:
    """教材のクライアントの既定のスキーム名 bearerAuth は、Agent Card の oauth2 と合わず、何も付かない。"""
    context = ClientCallContext(state={"sessionId": "default"})
    app, seen = build(AgentCard.model_validate(REQUIRED))
    interceptor = AuthInterceptor(await book_client_store("bearerAuth"))
    assert await send(app, [interceptor], context) == "completed"
    assert seen == [None]

    app, seen = build(AgentCard.model_validate(REQUIRED))
    interceptor = AuthInterceptor(await book_client_store("oauth2"))
    assert await send(app, [interceptor], context) == "completed"
    assert seen == [f"Bearer {TOKEN}"]


async def test_unauthorized_is_not_httpx_status_error() -> None:
    """401 は A2AClientHTTPError に包まれて届き、httpx.HTTPStatusError では捕まらない。"""
    app, _ = build(AgentCard.model_validate(REQUIRED), enforce=True)
    with pytest.raises(A2AClientHTTPError) as caught:
        await send(app)
    assert caught.value.status_code == 401
    assert not isinstance(caught.value, httpx.HTTPStatusError)


class FakeCerts:
    """Google の公開鍵の取得を、手元で作った証明書に差し替える。"""

    def __init__(self, cert_pem: str) -> None:
        self.cert_pem = cert_pem

    def __call__(self, url, method="GET", **kwargs):
        del url, method, kwargs
        data = json.dumps({"k1": self.cert_pem}).encode()
        return type("Response", (), {"status": 200, "data": data})()


def signer_and_certs() -> tuple[crypt.RSASigner, FakeCerts]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    signer = crypt.RSASigner.from_string(pem, key_id="k1")
    return signer, FakeCerts(cert.public_bytes(serialization.Encoding.PEM).decode())


def token(signer, aud: str, iss: str = "https://accounts.google.com") -> str:
    now = int(time.time())
    claims = {"iss": iss, "aud": aud, "sub": "x", "iat": now, "exp": now + 300}
    return jwt.encode(signer, claims).decode()


def test_verify_without_audience_accepts_any_audience() -> None:
    """OAUTH2_AUDIENCE が未設定だと audience が None になり、別のアプリ向けのトークンも通る。"""
    signer, certs = signer_and_certs()
    other = token(signer, aud="another-app")
    assert (
        id_token.verify_oauth2_token(other, certs, audience=None)["aud"]
        == "another-app"
    )
    with pytest.raises(ValueError):
        id_token.verify_oauth2_token(other, certs, audience="expense-agent")


def test_wrong_issuer_is_not_value_error() -> None:
    """発行者の誤りは GoogleAuthError で、教材の except ValueError では捕まらない。"""
    signer, certs = signer_and_certs()
    forged = token(signer, aud="expense-agent", iss="https://evil.example.com")
    with pytest.raises(GoogleAuthError) as caught:
        id_token.verify_oauth2_token(forged, certs, audience="expense-agent")
    assert not isinstance(caught.value, ValueError)


@pytest.fixture(scope="module")
def secured_go_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Go の ADK の A2A のサーバーを、OAuth 2 の宣言と Bearer の検証つきで起動する。"""
    from samples.a2a.test_a2a import GO_DIR, free_port

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
    env = os.environ | {"A2A_INTEROP_TOKEN": TOKEN}
    proc = subprocess.Popen([str(binary), "-addr", addr], env=env)
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


async def send_to(url: str, interceptors=None) -> str:
    client = await ClientFactory.connect(
        url, client_config=ClientConfig(streaming=False), interceptors=interceptors
    )
    async for event in client.send_message(
        create_text_message_object(content="交通費")
    ):
        if isinstance(event, tuple):
            return event[0].status.state.value
    return "none"


async def test_python_client_reads_go_requirement(secured_go_server: str) -> None:
    """Go の互換の Agent Card の OAuth 2 の要件を a2a-sdk 0.3 が読み、Bearer を付けて v0.3 の口を通る。"""
    async with httpx.AsyncClient() as hc:
        raw = (await hc.get(f"{secured_go_server}/.well-known/agent-card.json")).json()
    card = AgentCard.model_validate(raw)
    assert card.security == [{"oauth2": ["expense:read"]}]
    assert card.security_schemes["oauth2"].root.flows.client_credentials is not None

    with pytest.raises(A2AClientHTTPError) as caught:
        await send_to(secured_go_server)
    assert caught.value.status_code == 401
    assert await send_to(secured_go_server, [AuthInterceptor(Fixed())]) == "completed"
