"""A2A を mTLS でつなぎ、Agent Card の取得と呼び出しの両方にクライアント証明書が使われるかを確かめる。"""

from __future__ import annotations

import datetime
import ipaddress
import ssl
import threading
import time
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn
from a2a.client import ClientConfig, ClientFactory, create_text_message_object
from a2a.client.errors import A2AClientHTTPError
from a2a.types import AgentCapabilities, AgentCard
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from google.adk import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

from samples.a2a.test_a2a import free_port


class Ok(BaseLlm):
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="ok")])
        )


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _write(path: Path, key, cert: x509.Certificate) -> None:
    path.with_suffix(".key").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    path.with_suffix(".crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def make_pki(root: Path) -> None:
    """認証局、サーバー証明書（127.0.0.1）、クライアント証明書を作る。"""
    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (
        x509.CertificateBuilder()
        .subject_name(_name("test-ca"))
        .issuer_name(_name("test-ca"))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    _write(root / "ca", ca_key, ca)

    def leaf(cn: str, usage, san: list[x509.GeneralName] | None) -> None:
        key = ec.generate_private_key(ec.SECP256R1())
        builder = (
            x509.CertificateBuilder()
            .subject_name(_name(cn))
            .issuer_name(ca.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                critical=False,
            )
        )
        if san:
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san), critical=False
            )
        _write(root / cn, key, builder.sign(ca_key, hashes.SHA256()))

    leaf(
        "server",
        ExtendedKeyUsageOID.SERVER_AUTH,
        [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))],
    )
    leaf("client", ExtendedKeyUsageOID.CLIENT_AUTH, None)


@pytest.fixture(scope="module")
def mtls_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, Path]]:
    root = tmp_path_factory.mktemp("pki")
    make_pki(root)
    port = free_port()
    base = f"https://127.0.0.1:{port}"
    card = AgentCard(
        name="expense",
        description="経費",
        url=base,
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )
    app = to_a2a(
        Agent(name="expense", model=Ok(model="gemini-3.8-flash"), instruction="x"),
        agent_card=card,
    )
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        ssl_certfile=str(root / "server.crt"),
        ssl_keyfile=str(root / "server.key"),
        ssl_ca_certs=str(root / "ca.crt"),
        ssl_cert_reqs=ssl.CERT_REQUIRED,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        yield base, root
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def context(root: Path, with_client_cert: bool) -> ssl.SSLContext:
    """教材と同じ組み方。with_client_cert が False ならサーバーの検証だけをする。"""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if with_client_cert:
        ctx.load_cert_chain(certfile=root / "client.crt", keyfile=root / "client.key")
    ctx.load_verify_locations(cafile=root / "ca.crt")
    return ctx


async def call(base: str, ctx: ssl.SSLContext) -> str:
    async with httpx.AsyncClient(verify=ctx) as hc:
        client = await ClientFactory.connect(
            base, client_config=ClientConfig(httpx_client=hc, streaming=False)
        )
        async for event in client.send_message(
            create_text_message_object(content="経費")
        ):
            if isinstance(event, tuple):
                return event[0].status.state.value
    return "none"


async def test_client_certificate_reaches_card_and_call(
    mtls_server: tuple[str, Path],
) -> None:
    """クライアント証明書を持たせた httpx のクライアントで、Agent Card の取得から呼び出しまで通る。"""
    base, root = mtls_server
    assert await call(base, context(root, with_client_cert=True)) == "completed"


async def test_without_client_certificate_fails_at_card(
    mtls_server: tuple[str, Path],
) -> None:
    """クライアント証明書が無いと、Agent Card の取得の段階で失敗する。"""
    base, root = mtls_server
    with pytest.raises(A2AClientHTTPError) as caught:
        await call(base, context(root, with_client_cert=False))
    assert caught.value.status_code == 503
    assert "fetching agent card" in caught.value.message


async def test_client_config_without_httpx_client_ignores_tls(
    mtls_server: tuple[str, Path],
) -> None:
    """ClientConfig に httpx のクライアントを渡し忘れると、既定の検証で Agent Card の取得に失敗する。"""
    base, _ = mtls_server
    with pytest.raises(A2AClientHTTPError) as caught:
        await ClientFactory.connect(base, client_config=ClientConfig(streaming=False))
    assert caught.value.status_code == 503
    assert "CERTIFICATE_VERIFY_FAILED" in caught.value.message
