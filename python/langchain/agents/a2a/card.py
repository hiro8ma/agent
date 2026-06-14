"""AgentCard — an agent's self-description, served at ``.well-known/agent.json``.

MCP exposes *tools* a single agent reaches out to (an agent → tool edge); A2A
exposes *agents* that talk to each other as peers (an agent ↔ agent edge). The
card is the A2A discovery document: before any call, a client fetches it to learn
who the agent is, what it can do, and how to reach it — the agent's public API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Schema:
    """JSON-Schema-ish shape for one capability's input and output payloads."""

    input: dict[str, Any]
    output: dict[str, Any]


@dataclass(frozen=True)
class AgentCard:
    """The self-describing document a peer fetches to decide whether to talk to us.

    ``identity``     human/agent name.
    ``version``      semantic version; the client uses major for compat checks.
    ``capabilities`` named skills the agent advertises (the negotiation surface).
    ``schemas``      per-capability input/output contracts.
    ``endpoint``     where to send JSON-RPC calls (a dummy URL in this offline demo).
    ``auth_methods`` accepted auth schemes, e.g. ``["none", "bearer"]``.
    """

    identity: str
    version: str
    capabilities: list[str]
    schemas: dict[str, Schema]
    endpoint: str
    auth_methods: list[str] = field(default_factory=lambda: ["none"])

    def to_json(self) -> dict[str, Any]:
        """Render the card as the JSON a real ``.well-known/agent.json`` would serve."""
        return asdict(self)

    def major(self) -> int:
        return int(self.version.split(".", 1)[0])
