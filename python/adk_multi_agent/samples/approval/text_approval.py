"""教材の ApprovalManager と handle_approval_input を、省略された部分を最小限に補って再現する。

    承認は、利用者が「承認: {request_id}」と入力すると、before_model_callback が State にフラグを立てる
    承認待ちは _pending_approval の 1 枠に JSON で置く
    承認済みのフラグ _approval_{tool} は、次の呼び出しで 1 回だけ使って消す

確かめたこと（test_text_approval.py）

    承認するのは依頼した本人で、入力した文に ID が入っていれば通る
    承認の直後に「実行します」と返すが、モデルを呼ばないので実行されていない
    承認待ちが 2 件になると、先の依頼は上書きされて承認できなくなる
"""

from __future__ import annotations

import json
import uuid

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

APPROVAL_RULES = {
    "transfer_funds": {
        "condition": lambda args: args.get("amount", 0) >= 1_000_000,
        "message": "高額送金のため承認が必要です",
    },
}


def before_tool_callback(
    tool: BaseTool, args: dict, tool_context: ToolContext
) -> dict | None:
    rule = APPROVAL_RULES.get(tool.name)
    if not rule or not rule["condition"](args):
        return None
    approval_key = f"_approval_{tool.name}"
    if tool_context.state.get(approval_key):
        tool_context.state[approval_key] = False
        return None
    request_id = str(uuid.uuid4())[:8]
    approval_request = {
        "request_id": request_id,
        "tool_name": tool.name,
        "tool_input": args,
    }
    tool_context.state["_pending_approval"] = json.dumps(
        approval_request, ensure_ascii=False
    )
    return {
        "status": "approval_required",
        "request_id": request_id,
        "message": f"{rule['message']}。承認する場合は「承認: {request_id}」と入力してください。",
    }


def _get_last_user_message(llm_request: LlmRequest) -> str:
    for content in reversed(llm_request.contents or []):
        if content.role == "user":
            texts = [p.text for p in content.parts or [] if p.text]
            if texts:
                return "".join(texts)
    return ""


def _create_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def handle_approval_input(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    pending = callback_context.state.get("_pending_approval")
    if not pending:
        return None
    approval_request = json.loads(pending)
    last_message = _get_last_user_message(llm_request)
    if f"承認: {approval_request['request_id']}" in last_message:
        tool_name = approval_request["tool_name"]
        callback_context.state[f"_approval_{tool_name}"] = True
        callback_context.state["_pending_approval"] = None
        return _create_response(f"承認されました。{tool_name}を実行します。")
    return None
