from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as `uv run python bin/a2a.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.a2a import (  # noqa: E402
    A2AClient,
    A2AError,
    CapabilityMissingError,
    InProcessTransport,
    VersionIncompatibleError,
    build_server,
)

DEMO_TEXT = (
    "Agent-to-Agent lets autonomous agents discover one another and collaborate. "
    "Each agent publishes a card describing its capabilities. A client negotiates "
    "compatibility before issuing any call. This keeps integrations loosely coupled."
)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    transport = InProcessTransport(build_server())

    # (1) discovery — fetch the peer's .well-known/agent.json equivalent.
    section("1. card discovery")
    card = A2AClient(transport, speaks_major=1).discover()
    print(json.dumps(card.to_json(), indent=2))

    # (2a) handshake success — same major version, required capability present.
    section("2a. handshake (compatible)")
    client = A2AClient(transport, speaks_major=1)
    try:
        hs = client.handshake(needs=["summarizeText"])
        print(f"OK  peer={hs.card.identity} v{hs.card.version} granted={hs.granted}")
    except A2AError as exc:
        print(f"FAIL [{exc.code}] {exc.message}")

    # (2b) handshake failure — version mismatch.
    section("2b. handshake (version mismatch)")
    try:
        A2AClient(transport, speaks_major=2).handshake(needs=["summarizeText"])
        print("OK (unexpected)")
    except VersionIncompatibleError as exc:
        print(f"FAIL [{exc.code}] {exc.message}")

    # (2c) handshake failure — capability missing.
    section("2c. handshake (capability missing)")
    try:
        A2AClient(transport, speaks_major=1).handshake(needs=["translateText"])
        print("OK (unexpected)")
    except CapabilityMissingError as exc:
        print(f"FAIL [{exc.code}] {exc.message}")

    # (3) JSON-RPC call — invoke the negotiated capability.
    section("3. JSON-RPC summarizeText")
    response = client.call("summarizeText", {"text": DEMO_TEXT, "maxWords": 10})
    print(json.dumps(response, indent=2))

    # (4) unknown method — server returns JSON-RPC -32601.
    section("4. JSON-RPC unknown method")
    response = client.call("translateText", {"text": "hello"})
    print(json.dumps(response, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
