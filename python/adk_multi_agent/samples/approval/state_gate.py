"""教材の設計どおり、承認待ちと承認済みを State に置く before_tool_callback。

    APPROVAL_RULES でツールと条件を決める
    条件に当たれば request_id を作って承認待ちを State に置き、ツールを止める
    State に承認済みの印があれば通す

この形には抜け道が 2 つある。テスト（test_state_gate.py）で確かめる。

    State はクライアントが /run の state_delta で書ける。利用者が自分で承認済みにできる
    承認が引数に結び付いていない。100 万円で承認させた後に 1,000 万円で呼び直せる
"""

from __future__ import annotations

import uuid
from typing import Any

from google.adk.tools import BaseTool, ToolContext

APPROVAL_RULES = {
    "transfer_funds": lambda args: args.get("amount", 0) >= 1_000_000,
}

EXECUTED: list[dict] = []


def transfer_funds(amount: int, to: str) -> dict:
    """送金する。

    Args:
        amount: 金額（円）
        to: 送金先の口座
    """
    EXECUTED.append({"amount": amount, "to": to})
    return {"status": "sent", "amount": amount, "to": to}


def approval_key(tool_name: str) -> str:
    return f"approval:{tool_name}"


def require_approval(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict | None:
    rule = APPROVAL_RULES.get(tool.name)
    if rule is None or not rule(args):
        return None
    if tool_context.state.get(approval_key(tool.name)) == "approved":
        return None
    request_id = uuid.uuid4().hex[:8]
    tool_context.state["approval:pending"] = {
        "request_id": request_id,
        "tool": tool.name,
        "args": args,
    }
    return {"status": "pending_approval", "request_id": request_id}
