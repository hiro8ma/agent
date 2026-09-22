"""同じツールを同じ引数で呼び続けるループを止めるプラグイン。

ADK の max_llm_calls はモデルの呼び出しの総数だけを見るので、既定の 500 回に達するまで同じ呼び出しが続く。
このプラグインは Invocation ごとに (ツール名, 引数) を数え、limit を超えた呼び出しを実行せずに打ち切る。
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


class RepeatedToolCallGuard(BasePlugin):
    def __init__(self, limit: int = 3) -> None:
        super().__init__(name="repeated_tool_call_guard")
        self.limit = limit
        self._counts: dict[str, Counter[str]] = {}

    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict | None:
        key = f"{tool.name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)}"
        counts = self._counts.setdefault(tool_context.invocation_id, Counter())
        counts[key] += 1
        if counts[key] <= self.limit:
            return None
        # 以後のモデルの呼び出しも止めるため、Invocation を終わらせる。
        tool_context._invocation_context.end_invocation = True
        return {
            "status": "error",
            "error_message": f"{tool.name} を同じ引数で {self.limit} 回呼んだので打ち切った",
        }
