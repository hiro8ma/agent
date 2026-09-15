"""ガードレールを 3 層に置く。

    入力   before_model   インジェクションの兆候を見て、モデルへ渡さない
    出力   after_model    個人情報の形を見て、伏せてから返す
    実行   before_tool    壊す操作に確認を挟み、確認済みでなければ実行しない

いずれも第一防御線であって、これだけで安全にはならない。
パターンで拾えるのは既知の形だけなので、分類器や権限設計と併用する。

検査は実物の型で書く。`LlmRequest.contents` は `list[Content]` なので、
MagicMock で代用すると形の誤りが隠れる。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

# 検出したことを残す鍵。監視と、後段での扱いを変える判断に使う。
INJECTION_FLAG_KEY = "injection_detected"
INJECTION_PATTERN_KEY = "injection_pattern"

# 確認済みかどうかの鍵。確認は 1 回のやり取りで閉じるので temp: にしない。
CONFIRMED_KEY = "confirmed_destructive"

INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(the\s+)?(system|previous)\s+(prompt|instructions)",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"以前の指示を(すべて)?無視",
    r"これまでの指示を(すべて)?無視",
    r"システムプロンプトを(表示|教えて|出力)",
    r"あなたの指示を(表示|教えて|出力)",
)

# 出力に混ざってはいけない形。伏せ字は何を伏せたか分かる名前にする。
PII_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL_MASKED]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD_MASKED]"),
    (re.compile(r"\b0\d{1,4}-\d{1,4}-\d{4}\b"), "[PHONE_MASKED]"),
)


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def last_user_text(request: LlmRequest) -> str:
    """直近の利用者発話だけを取り出す。

    全履歴を対象にすると、過去に弾いた入力で毎回落ちる。
    """
    for content in reversed(request.contents or []):
        if content.role != "user":
            continue
        return "\n".join(
            part.text
            for part in content.parts or []
            if part.text and not getattr(part, "thought", False)
        )
    return ""


def detect_prompt_injection(
    patterns: Sequence[str] = INJECTION_PATTERNS,
    message: str = "ご質問の内容をもう少し具体的にお教えいただけますか。",
):
    """入力ガードレール。兆候があればモデルを呼ばずに返す。"""
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    def callback(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        text = last_user_text(llm_request)
        for pattern in compiled:
            if pattern.search(text):
                callback_context.state[INJECTION_FLAG_KEY] = True
                callback_context.state[INJECTION_PATTERN_KEY] = pattern.pattern
                return _text_response(message)
        return None

    return callback


def mask_pii(rules: Sequence[tuple[re.Pattern[str], str]] = PII_RULES):
    """出力ガードレール。個人情報の形を伏せる。

    伏せたときだけ差し替えた応答を返す。変更が無ければ None を返して元の応答を通す。
    """

    def callback(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if llm_response.content is None:
            return None
        changed = False
        parts: list[types.Part] = []
        for part in llm_response.content.parts or []:
            if not part.text:
                parts.append(part)
                continue
            text = part.text
            for pattern, placeholder in rules:
                text, hit = pattern.subn(placeholder, text)
                changed = changed or bool(hit)
            parts.append(types.Part(text=text))
        if not changed:
            return None
        return LlmResponse(
            content=types.Content(
                role=llm_response.content.role or "model", parts=parts
            )
        )

    return callback


def require_confirmation(
    destructive_tools: Sequence[str], flag_key: str = CONFIRMED_KEY
):
    """実行ガードレール。壊す操作は確認済みでなければ実行しない。

    返すのは中身のある dict にする。空の dict でもツールは飛ぶが、
    理由が残らないのでモデルは何を聞き返せばよいか分からない。
    """
    targets = set(destructive_tools)

    def callback(tool: BaseTool, args: dict, tool_context: ToolContext) -> dict | None:
        if tool.name not in targets:
            return None
        if tool_context.state.get(flag_key):
            return None
        return {
            "status": "confirmation_required",
            "tool": tool.name,
            "args": args,
            "message": f"{tool.name} は取り消せない操作です。実行してよいか確認してください",
        }

    return callback
