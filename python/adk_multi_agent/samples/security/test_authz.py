"""認証・認可の教材のコード例が前提にしていることを確かめる。"""

from __future__ import annotations

import json
import ssl
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jwt
import pytest
from a2a.types import AgentCard, MutualTLSSecurityScheme, OAuth2SecurityScheme
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt import PyJWKClient

ISSUER = "https://auth.example.com"
AUDIENCE = "customer-agent"


@pytest.fixture
def idp() -> Iterator[tuple[str, rsa.RSAPrivateKey, list[str]]]:
    """JWKS を配り、取りに来た回数を数える。"""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update({"kid": "k1", "use": "sig", "alg": "RS256"})
    body = json.dumps({"keys": [jwk]}).encode()
    fetched: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            fetched.append(self.path)
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/jwks.json", key, fetched
    finally:
        server.shutdown()


def token(key: rsa.RSAPrivateKey, scope: str) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "billing-agent",
        "iat": now,
        "exp": now + 300,
        "scope": scope,
    }
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})


def verify_a2a_token(
    tok: str, jwks_url: str, expected_audience: str, expected_issuer: str
) -> dict:
    """教材の検証。呼び出しのたびに PyJWKClient を作る。"""
    signing_key = PyJWKClient(jwks_url).get_signing_key_from_jwt(tok)
    return jwt.decode(
        tok,
        signing_key.key,
        algorithms=["RS256"],
        audience=expected_audience,
        issuer=expected_issuer,
    )


def test_book_verifier_fetches_jwks_on_every_call(idp) -> None:
    """教材の書き方では、要求のたびに JWKS を取りに行く。クライアントを使い回せば 1 回で済む。"""
    url, key, fetched = idp
    tok = token(key, "customer.read")
    for _ in range(5):
        verify_a2a_token(tok, url, AUDIENCE, ISSUER)
    assert len(fetched) == 5

    fetched.clear()
    shared = PyJWKClient(url)
    for _ in range(5):
        jwt.decode(
            tok,
            shared.get_signing_key_from_jwt(tok).key,
            algorithms=["RS256"],
            audience=AUDIENCE,
            issuer=ISSUER,
        )
    assert len(fetched) == 1


def test_book_verifier_does_not_check_scope(idp) -> None:
    """署名、宛先、発行者が正しければ、スコープが customer.read だけでも検証は通る。書き込みを止めるのは呼び出し側の責任。"""
    url, key, _ = idp
    claims = verify_a2a_token(token(key, "customer.read"), url, AUDIENCE, ISSUER)
    assert "customer.write" not in claims["scope"].split()


def test_book_agent_card_parses_as_v03() -> None:
    """教材の Agent Card は v0.3 の security のキーで書かれ、本文の説明（v1.0 の securityRequirements）と食い違う。"""
    book = {
        "name": "customer",
        "description": "顧客",
        "url": "https://customer.example.com",
        "version": "1.0.0",
        "capabilities": {},
        "default_input_modes": ["text"],
        "default_output_modes": ["text"],
        "skills": [],
        "securitySchemes": {
            "corp_oauth": {
                "type": "oauth2",
                "flows": {
                    "clientCredentials": {
                        "tokenUrl": "https://auth.example.com/oauth/token",
                        "scopes": {"customer.read": "参照"},
                    }
                },
            },
            "agent_mtls": {"type": "mutualTLS"},
        },
        "security": [{"corp_oauth": ["customer.read"], "agent_mtls": []}],
    }
    parsed = AgentCard.model_validate(book)
    assert isinstance(parsed.security_schemes["corp_oauth"].root, OAuth2SecurityScheme)
    assert isinstance(
        parsed.security_schemes["agent_mtls"].root, MutualTLSSecurityScheme
    )
    assert parsed.security == [{"corp_oauth": ["customer.read"], "agent_mtls": []}]


def test_client_context_already_requires_server_cert() -> None:
    """PROTOCOL_TLS_CLIENT は最初からサーバーの証明書の検証を必須にする。クライアントの証明書を必須にするのはサーバー側の設定。"""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    assert server.verify_mode == ssl.CERT_NONE
