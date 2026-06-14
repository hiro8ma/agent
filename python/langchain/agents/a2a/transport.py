"""Transport abstraction — how a client reaches a server.

Real A2A speaks HTTP: GET ``.well-known/agent.json`` for the card, POST JSON-RPC
for calls (e.g. ``http://localhost:PORT`` in production). This demo keeps the same
two operations but binds them to an in-process server object, so the protocol shape
is exercised with zero network and zero extra dependencies. Swap this class for an
HTTP-backed one and the client/handshake code is unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol

from .card import AgentCard
from .server import A2AServer


class Transport(Protocol):
    """The two A2A operations a client needs, independent of the wire."""

    def fetch_card(self) -> AgentCard: ...

    def call(self, request: dict[str, Any]) -> dict[str, Any]: ...


class InProcessTransport:
    """Routes card fetches and JSON-RPC calls straight to a server in memory."""

    def __init__(self, server: A2AServer) -> None:
        self._server = server

    def fetch_card(self) -> AgentCard:
        return self._server.get_card()

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._server.dispatch(request)
