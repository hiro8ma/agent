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
from .semantic import ToolMatch, select_tools_semantically


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


# Registry of wired mcp/ servers: name -> (server_dir, entrypoint).
# Each is its own uv project, launched with `uv run --directory <dir> python <entry>`.
_SERVERS: dict[str, tuple[str, str]] = {
    "calc": ("calc", "calculator_server.py"),
    "memory": ("memory", "memory_server.py"),
}


def build_mcp_client(
    mcp_repo_path: str | None = None,
    servers: list[str] | None = None,
) -> MultiServerMCPClient:
    """Build a MultiServerMCPClient wired to selected mcp/ servers over stdio.

    ``servers`` selects which entries of ``_SERVERS`` to wire (default: calc).
    Tools from every wired server share one namespace, so the agent picks tools
    by name (e.g. ``remember`` / ``recall`` from memory, arithmetic from calc).
    """

    repo = resolve_mcp_repo(mcp_repo_path)
    selected = servers or ["calc"]

    def uv_stdio(server_dir: str, entrypoint: str) -> dict[str, Any]:
        return {
            "command": "uv",
            "args": ["run", "--directory", str(repo / server_dir), "python", entrypoint],
            "transport": "stdio",
        }

    connections: dict[str, dict[str, Any]] = {}
    for name in selected:
        if name not in _SERVERS:
            raise ValueError(
                f"unknown mcp server '{name}'. known: {sorted(_SERVERS)}"
            )
        server_dir, entrypoint = _SERVERS[name]
        connections[name] = uv_stdio(server_dir, entrypoint)

    return MultiServerMCPClient(connections)


async def call_tool(
    tool_name: str,
    args: dict[str, Any],
    mcp_repo_path: str | None = None,
    servers: list[str] | None = None,
) -> str:
    """Invoke a single MCP tool by name and return its string output.

    Lets callers drive the remember/recall loop without an LLM (key-free path).
    """

    client = build_mcp_client(mcp_repo_path, servers)
    tools: list[BaseTool] = await client.get_tools()
    for tool in tools:
        if tool.name == tool_name:
            return _stringify_tool_result(await tool.ainvoke(args))
    available = ", ".join(sorted(t.name for t in tools))
    raise ValueError(f"tool '{tool_name}' not found. available: {available}")


def _stringify_tool_result(result: Any) -> str:
    """Flatten an MCP tool result (str or list of content blocks) to text."""

    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for part in result:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "\n".join(parts)
    return str(result)


async def list_tools(
    mcp_repo_path: str | None = None, servers: list[str] | None = None
) -> list[tuple[str, str]]:
    """Return (name, description) for every tool exposed by the MCP servers."""

    client = build_mcp_client(mcp_repo_path, servers)
    tools = await client.get_tools()
    return [(t.name, t.description or "") for t in tools]


async def run(
    question: str,
    mcp_repo_path: str | None = None,
    servers: list[str] | None = None,
) -> str:
    """Answer a question, letting the LLM call MCP tools as needed.

    Pipeline: connect to mcp/ servers over stdio -> get_tools() -> bind tools to a
    LangGraph create_react_agent -> invoke -> extract final assistant message.
    """

    client = build_mcp_client(mcp_repo_path, servers)
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


async def select_tools_for_query(
    question: str,
    k: int = 3,
    mcp_repo_path: str | None = None,
    servers: list[str] | None = None,
) -> tuple[list[BaseTool], list[ToolMatch]]:
    """Fetch all MCP tools, then semantically narrow to the top-k for a query.

    Returns ``(all_tools, matches)`` so callers can show the tool-RAG reduction
    (``len(all_tools)`` -> ``len(matches)``) before binding.
    """

    client = build_mcp_client(mcp_repo_path, servers)
    all_tools: list[BaseTool] = await client.get_tools()
    matches = select_tools_semantically(question, all_tools, k=k)
    return all_tools, matches


async def run_semantic(
    question: str,
    k: int = 3,
    mcp_repo_path: str | None = None,
    servers: list[str] | None = None,
) -> str:
    """Answer a question after binding only the top-k semantically selected tools.

    Same pipeline as ``run`` but with a tool-RAG retrieval step in front, so the
    LLM sees ``k`` tools instead of every tool the servers expose.
    """

    _, matches = await select_tools_for_query(question, k, mcp_repo_path, servers)
    selected = [m.tool for m in matches]

    agent = build_agent(
        provider=select_provider(),
        system_prompt=MCP_CLIENT_SYSTEM_PROMPT,
        tools=selected,
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
