"""Typed errors for A2A negotiation and JSON-RPC dispatch.

JSON-RPC reserves codes in the -32700..-32600 range; A2A-level negotiation
failures (version / capability mismatch) use an application range below that so a
client can tell "we never agreed to talk" apart from "the call itself failed".
"""

from __future__ import annotations

# JSON-RPC 2.0 reserved codes.
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

# A2A handshake (application-defined) codes.
VERSION_INCOMPATIBLE = -33001
CAPABILITY_MISSING = -33002


class A2AError(Exception):
    """Base for A2A failures; carries a JSON-RPC-shaped code + message."""

    code: int = -32000

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code

    def to_rpc_error(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message}


class VersionIncompatibleError(A2AError):
    code = VERSION_INCOMPATIBLE


class CapabilityMissingError(A2AError):
    code = CAPABILITY_MISSING


class MethodNotFoundError(A2AError):
    code = METHOD_NOT_FOUND
