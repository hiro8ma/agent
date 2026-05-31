from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

DEFAULT_WORKSPACE = "workspace"


class WriteFileInput(BaseModel):
    """Input schema for the write_file tool."""

    relative_path: str = Field(
        ...,
        description=(
            "Destination path relative to the agent workspace, e.g. 'report.html'. "
            "Absolute paths and parent-directory escapes ('..') are rejected."
        ),
    )
    content: str = Field(
        ...,
        description=(
            "Complete file contents to write, as UTF-8 text. "
            "Pass the entire final document; the write overwrites the target, "
            "it does not append."
        ),
    )


def _resolve_within_workspace(workspace: Path, relative_path: str) -> Path:
    """Resolve relative_path under workspace, refusing any escape outside it.

    Blocks absolute paths and '..' traversal so a tool call cannot clobber files
    elsewhere on the host. This is the confinement half of the HITL contract:
    the human approves, but the agent still cannot reach beyond the workspace.
    """

    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {relative_path!r}")

    target = (workspace / candidate).resolve()
    root = workspace.resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"path escapes the workspace: {relative_path!r}")
    return target


def build_write_file_tool(
    workspace: str | Path = DEFAULT_WORKSPACE,
    name: str = "write_file",
    description: str = (
        "Write text or HTML to a file inside the agent workspace. "
        "This is a destructive operation: it requires explicit human approval "
        "before the bytes hit disk. Input is a workspace-relative path and the content."
    ),
) -> BaseTool:
    """Wrap a confined file write as a HITL-gated StructuredTool.

    Before writing, the tool calls langgraph.types.interrupt() with a preview of
    the action. The graph pauses; the caller resumes with Command(resume=...).
    'approve' performs the write, anything else skips it. The graph must run with
    a checkpointer (see cli.agent.build_hitl_agent) for interrupt/resume to work.
    """

    workspace_path = Path(workspace)

    def _run(relative_path: str, content: str) -> str:
        target = _resolve_within_workspace(workspace_path, relative_path)

        decision = interrupt(
            {
                "action": "write_file",
                "path": str(target),
                "bytes": len(content.encode("utf-8")),
                "preview": content[:500],
            }
        )

        approved = decision is True or (
            isinstance(decision, str) and decision.strip().lower() == "approve"
        )
        if not approved:
            return f"write_file denied by human; {target} left unchanged."

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content.encode('utf-8'))} bytes to {target}"

    return StructuredTool.from_function(
        name=name,
        description=description,
        func=_run,
        args_schema=WriteFileInput,
    )
