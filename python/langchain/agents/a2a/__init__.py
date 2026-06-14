"""a2a — an offline, dependency-free demo of the Agent-to-Agent protocol.

A2A lets autonomous agents discover and call *each other* as peers, where MCP lets
one agent reach *tools*. The flow modelled here: an agent publishes an
``AgentCard`` (its ``.well-known/agent.json`` self-description) → a client fetches
it and negotiates compatibility (version + capabilities) → on agreement the client
dispatches work over JSON-RPC 2.0.

Everything runs in-process: ``InProcessTransport`` stands in for HTTP so the card →
handshake → JSON-RPC structure is exercised with stdlib only and zero network.

``card.py``       — AgentCard / Schema, the discovery document.
``server.py``     — A2AServer: serves the card and dispatches JSON-RPC to handlers.
``client.py``     — A2AClient: discover → handshake → call_result.
``transport.py``  — Transport protocol + in-process implementation.
``summarizer.py`` — SummarizerAgent sample (deterministic offline summary).
``errors.py``     — typed handshake + JSON-RPC errors.
"""

from .card import AgentCard, Schema
from .client import A2AClient, Handshake
from .errors import (
    A2AError,
    CapabilityMissingError,
    MethodNotFoundError,
    VersionIncompatibleError,
)
from .server import A2AServer
from .summarizer import CAPABILITY, build_server
from .transport import InProcessTransport, Transport

__all__ = [
    "CAPABILITY",
    "A2AClient",
    "A2AError",
    "A2AServer",
    "AgentCard",
    "CapabilityMissingError",
    "Handshake",
    "InProcessTransport",
    "MethodNotFoundError",
    "Schema",
    "Transport",
    "VersionIncompatibleError",
    "build_server",
]
