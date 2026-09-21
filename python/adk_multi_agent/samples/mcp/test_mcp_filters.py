"""教材の MCP のツールの絞り込み、パスの検査、接続の後片付けを確かめる。MCP サーバーは起動しない。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_toolset import BaseToolset

TOOLS = json.loads((Path(__file__).parent / "filesystem_tools.json").read_text())[
    "tools"
]


def safe_tools_only(tool, readonly_context=None):
    """教材のフィルタ。名前に書き込み系のキーワードを含むツールを除く。"""
    write_keywords = ["write", "delete", "update", "insert", "drop", "create"]
    return not any(kw in tool.name.lower() for kw in write_keywords)


async def handle_tool_errors(tool, args: dict, tool_context=None):
    """教材の before_tool_callback。path の引数だけを見る。"""
    if "path" in args:
        path = args["path"]
        if ".." in path or path.startswith("/"):
            return {"error": "指定されたパスにはアクセスできません"}
    return None


def test_keyword_filter_keeps_destructive_tools() -> None:
    """名前に書き込み系の語が無い edit_file と move_file が、破壊的なのに残る。"""
    kept = {
        t["name"] for t in TOOLS if safe_tools_only(SimpleNamespace(name=t["name"]))
    }
    destructive_kept = sorted(
        n for n in kept if not next(t for t in TOOLS if t["name"] == n)["readOnlyHint"]
    )
    assert destructive_kept == ["edit_file", "move_file"]

    by_annotation = {t["name"] for t in TOOLS if t["readOnlyHint"]}
    assert "edit_file" not in by_annotation and "move_file" not in by_annotation


@pytest.mark.parametrize(
    ("args", "blocked"),
    [
        ({"path": "../../etc/passwd"}, True),
        ({"source": "../../etc/passwd", "destination": "stolen.txt"}, False),
        ({"paths": ["../../etc/passwd"]}, False),
        ({"path": "/srv/allowed/report.txt"}, True),
        ({"path": "notes..txt"}, True),
    ],
    ids=[
        "path の .. は止める",
        "move_file の source は見ていない",
        "read_multiple_files の paths は見ていない",
        "許可したディレクトリの絶対パスまで止める",
        "名前に .. を含むだけのファイルも止める",
    ],
)
async def test_path_check_only_sees_the_path_argument(
    args: dict, blocked: bool
) -> None:
    result = await handle_tool_errors(SimpleNamespace(name="x"), args)
    assert (result is not None) == blocked


class CountingToolset(BaseToolset):
    def __init__(self, name: str) -> None:
        super().__init__(tool_name_prefix=name)
        self.closed = False

    async def get_tools(self, readonly_context=None):
        def read_file(path: str) -> dict:
            """ファイルを読む。"""
            return {}

        return [FunctionTool(read_file)]

    async def close(self) -> None:
        self.closed = True


async def test_runner_close_skips_toolsets_inside_agent_tool() -> None:
    """Runner は tools とサブエージェントのツールセットを閉じるが、AgentTool の中までは辿らない。"""
    direct, in_sub, in_agent_tool = (
        CountingToolset("fs"),
        CountingToolset("gh"),
        CountingToolset("bq"),
    )
    sub = Agent(name="sub", model="gemini-3.8-flash", instruction="x", tools=[in_sub])
    wrapped = Agent(
        name="wrapped", model="gemini-3.8-flash", instruction="x", tools=[in_agent_tool]
    )
    root = Agent(
        name="root",
        model="gemini-3.8-flash",
        instruction="x",
        tools=[direct, AgentTool(agent=wrapped)],
        sub_agents=[sub],
    )
    runner = InMemoryRunner(agent=root, app_name="mcp")
    await runner.close()

    assert (direct.closed, in_sub.closed, in_agent_tool.closed) == (True, True, False)


async def test_prefix_joins_with_underscore() -> None:
    tools = await CountingToolset("fs").get_tools_with_prefix()
    assert [t.name for t in tools] == ["fs_read_file"]
