from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from cli.agent import build_agent
from core.providers.factory import select_provider

from .prompt import MCP_CLIENT_SYSTEM_PROMPT


def _default_mcp_repo_path() -> str:
    """Locate the sibling mcp/ repository.

    agent/ and mcp/ live under the same parent. Walk up until a directory whose
    sibling mcp/ contains the known server entrypoints is found.
    """

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.parent / "mcp"
        if (candidate / "calc" / "calculator_server.py").is_file():
            return str(candidate)
    return ""


def resolve_mcp_repo(mcp_repo_path: str | None = None) -> Path:
    """Resolve and validate the mcp/ repository path.

    Priority: explicit arg -> MCP_REPO_PATH env -> sibling autodetection.
    """

    repo = mcp_repo_path or os.environ.get("MCP_REPO_PATH") or _default_mcp_repo_path()
    if not repo:
        raise RuntimeError(
            "mcp/ repository not found. Set MCP_REPO_PATH to the directory that "
            "contains calc/calculator_server.py."
        )
    path = Path(repo).expanduser().resolve()
    if not (path / "calc" / "calculator_server.py").is_file():
        raise RuntimeError(
            f"MCP_REPO_PATH does not look like the mcp/ repo: {path} "
            "(expected calc/calculator_server.py)."
        )
    return path


def build_mcp_client(mcp_repo_path: str | None = None) -> MultiServerMCPClient:
    """Build a MultiServerMCPClient wired to selected mcp/ servers over stdio.

    Each server is its own uv project, so it is launched with
    `uv run --directory <server_dir> python <entrypoint>`. Only read-only,
    key-free servers are wired here to keep the integration side-effect free.
    """

    repo = resolve_mcp_repo(mcp_repo_path)

    def uv_stdio(server_dir: str, entrypoint: str) -> dict[str, Any]:
        return {
            "command": "uv",
            "args": ["run", "--directory", str(repo / server_dir), "python", entrypoint],
            "transport": "stdio",
        }

    return MultiServerMCPClient(
        {
            "calc": uv_stdio("calc", "calculator_server.py"),
        }
    )


async def list_tools(mcp_repo_path: str | None = None) -> list[tuple[str, str]]:
    """Return (name, description) for every tool exposed by the MCP servers."""

    client = build_mcp_client(mcp_repo_path)
    tools = await client.get_tools()
    return [(t.name, t.description or "") for t in tools]


async def run(question: str, mcp_repo_path: str | None = None) -> str:
    """Answer a question, letting the LLM call MCP tools as needed.

    Pipeline: connect to mcp/ servers over stdio -> get_tools() -> bind tools to a
    LangGraph create_react_agent -> invoke -> extract final assistant message.
    """

    client = build_mcp_client(mcp_repo_path)
    tools: list[BaseTool] = await client.get_tools()

    agent = build_agent(
        provider=select_provider(),
        system_prompt=MCP_CLIENT_SYSTEM_PROMPT,
        tools=tools,
    )

    result: dict[str, Any] = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return _extract_final_answer(result)


def _extract_final_answer(result: dict[str, Any]) -> str:
    """Pull the last assistant message content out of the agent state."""

    messages: list[BaseMessage] = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                if parts:
                    return "\n".join(parts)
    return "(agent produced no assistant message)"
