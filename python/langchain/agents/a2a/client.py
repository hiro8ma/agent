"""A2AClient — discover a peer, negotiate, then call it over JSON-RPC.

The handshake is the heart of A2A: before trusting a peer, the client fetches its
card and checks (1) the major version matches what it speaks and (2) every
capability it needs is advertised. Only then does it issue JSON-RPC calls. This is
what makes agents composable without prior hard-coding — they agree at runtime.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .card import AgentCard
from .errors import (
    A2AError,
    CapabilityMissingError,
    VersionIncompatibleError,
)
from .transport import Transport


@dataclass(frozen=True)
class Handshake:
    """The agreed-upon view after a successful negotiation."""

    card: AgentCard
    granted: list[str]


class A2AClient:
    """Speaks a given major version and negotiates with a peer before calling it."""

    def __init__(self, transport: Transport, speaks_major: int) -> None:
        self._transport = transport
        self._speaks_major = speaks_major
        self._ids: Iterator[int] = itertools.count(1)

    def discover(self) -> AgentCard:
        """Fetch the peer's self-description (the discovery step)."""
        return self._transport.fetch_card()

    def handshake(self, needs: list[str]) -> Handshake:
        """Negotiate: version-compat then capability presence. Raise on mismatch."""
        card = self.discover()

        if card.major() != self._speaks_major:
            raise VersionIncompatibleError(
                f"peer speaks v{card.version} (major {card.major()}), "
                f"client speaks major {self._speaks_major}"
            )

        missing = [c for c in needs if c not in card.capabilities]
        if missing:
            raise CapabilityMissingError(
                f"peer lacks required capabilities: {missing} "
                f"(advertised: {card.capabilities})"
            )
        return Handshake(card=card, granted=list(needs))

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Issue one JSON-RPC 2.0 request and return the raw response envelope."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": next(self._ids),
        }
        return self._transport.call(request)

    def call_result(self, method: str, params: dict[str, Any]) -> Any:
        """Call and unwrap ``result``, raising ``A2AError`` on a JSON-RPC error."""
        response = self.call(method, params)
        if "error" in response:
            err = response["error"]
            raise A2AError(str(err.get("message")), code=int(err.get("code")))
        return response["result"]
