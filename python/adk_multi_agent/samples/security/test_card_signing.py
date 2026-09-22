"""A2A の Agent Card の署名が、既定で検証されるか、どう検証すれば偽装を防げるかを確かめる。"""

from __future__ import annotations

import base64
import json

import httpx
import jwt
import pytest
from a2a.client import A2ACardResolver
from a2a.types import AgentCapabilities, AgentCard
from a2a.utils.signing import (
    InvalidSignaturesError,
    NoSignatureError,
    create_agent_card_signer,
    create_signature_verifier,
)
from cryptography.hazmat.primitives.asymmetric import ec
from google.adk import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from jwt.api_jwk import PyJWK

BASE = "http://inventory.local"


def card(url: str = BASE) -> AgentCard:
    return AgentCard(
        name="inventory",
        description="在庫",
        url=url,
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )


def keypair() -> tuple[ec.EllipticCurvePrivateKey, dict]:
    key = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(key.public_key()))
    return key, jwk


async def resolve(app, verifier=None) -> AgentCard:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as hc,
    ):
        return await A2ACardResolver(hc, BASE).get_agent_card(
            signature_verifier=verifier
        )


def serve(c: AgentCard):
    return to_a2a(
        Agent(name="inventory", model="gemini-3.8-flash", instruction="x"), agent_card=c
    )


async def test_unsigned_card_is_accepted_by_default() -> None:
    """検証の関数を渡さなければ、署名の無い Agent Card をそのまま受け取る（ADK の RemoteA2aAgent も渡さない）。"""
    got = await resolve(serve(card()))
    assert got.signatures is None


async def test_pinned_key_accepts_signed_and_rejects_unsigned_or_tampered() -> None:
    owner_key, owner_jwk = keypair()
    signer = create_agent_card_signer(owner_key, {"alg": "ES256", "kid": "owner"})
    pinned = {"owner": PyJWK(owner_jwk)}
    verifier = create_signature_verifier(lambda kid, jku: pinned[kid], ["ES256"])

    signed = signer(card())
    assert (await resolve(serve(signed), verifier)).signatures

    with pytest.raises(NoSignatureError):
        await resolve(serve(card()), verifier)

    tampered = signer(card())
    tampered.url = "http://attacker.local"
    with pytest.raises(InvalidSignaturesError):
        await resolve(serve(tampered), verifier)


async def test_trusting_jku_lets_an_attacker_sign_their_own_card() -> None:
    """ヘッダーの jku（鍵の置き場所）をそのまま信じて鍵を取ると、攻撃者の鍵で署名した Agent Card が通る。"""
    attacker_key, attacker_jwk = keypair()
    jwks_by_url = {"http://attacker.local/jwks.json": attacker_jwk}
    trusting = create_signature_verifier(
        lambda kid, jku: PyJWK(jwks_by_url[jku]), ["ES256"]
    )
    forged = create_agent_card_signer(
        attacker_key,
        {"alg": "ES256", "kid": "k1", "jku": "http://attacker.local/jwks.json"},
    )(card(url="http://attacker.local"))
    assert (await resolve(serve(forged), trusting)).url == "http://attacker.local"


def test_signer_defaults_to_hs256() -> None:
    """alg を指定しないと HS256（共有の秘密）で署名する。検証できる相手は署名もできる。"""
    signed = create_agent_card_signer("shared-secret", {"kid": "k1"})(card())
    header = json.loads(base64.urlsafe_b64decode(signed.signatures[0].protected + "=="))
    assert header["alg"] == "HS256"
