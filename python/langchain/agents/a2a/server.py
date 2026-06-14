"""A2AServer — publishes a card and dispatches JSON-RPC calls to capabilities.

A capability is a plain ``(params) -> result`` handler keyed by the method name a
client sends. ``dispatch`` is transport-agnostic: it takes a parsed JSON-RPC
request dict and returns a JSON-RPC response dict, so the same server works behind
an in-process call, an HTTP handler, or a queue worker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .card import AgentCard
from .errors import A2AError, MethodNotFoundError

Capability = Callable[[dict[str, Any]], Any]


class A2AServer:
    """Hosts one agent: serves its card and routes methods to capability handlers."""

    def __init__(self, card: AgentCard, handlers: dict[str, Capability]) -> None:
        advertised = set(card.capabilities)
        registered = set(handlers)
        if advertised != registered:
            # The card is a contract: every advertised capability must be runnable,
            # and we refuse to host a handler we did not advertise.
            raise ValueError(
                f"card/handler mismatch: advertised={sorted(advertised)} "
                f"registered={sorted(registered)}"
            )
        self._card = card
        self._handlers = handlers

    def get_card(self) -> AgentCard:
        """Serve the discovery document (the ``.well-known/agent.json`` fetch)."""
        return self._card

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run one JSON-RPC 2.0 request and return its JSON-RPC response."""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {}) or {}

        handler = self._handlers.get(method)
        if handler is None:
            err = MethodNotFoundError(f"Method not found: {method!r}")
            return self._error(req_id, err)

        try:
            result = handler(params)
        except A2AError as exc:
            return self._error(req_id, exc)
        return {"jsonrpc": "2.0", "result": result, "id": req_id}

    @staticmethod
    def _error(req_id: Any, exc: A2AError) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "error": exc.to_rpc_error(), "id": req_id}
