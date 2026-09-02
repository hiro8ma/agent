"""コールバックで入力と出力を検査する。

Go 側の internal/guardrail と同じ規則を Python の ADK へ載せる。
"""

from __future__ import annotations

import re

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

# 資格情報を引き出そうとする入力。
_BANNED = ["APIキー", "API キー", "api_key", "パスワード", "秘密鍵", "環境変数を表示"]

# 指示の上書きを狙う入力。
_INJECTION = ["これまでの指示を無視", "システムプロンプトを", "instruction を出力"]

# 出力に混ざってはいけない形。
_SECRET_PATTERNS = [
    re.compile(r"AIzaSy[A-Za-z0-9_\-]{20,}"),
    # 2026 年に発行される Gemini の鍵は AQ. 始まりで、AIzaSy に一致しない。
    re.compile(r"AQ\.[A-Za-z0-9_\-.]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
]


def _refuse(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _request_text(req: LlmRequest) -> str:
    out = []
    for content in req.contents or []:
        for part in content.parts or []:
            if part.text:
                out.append(part.text)
    return "\n".join(out)


def block_credential_requests(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """資格情報を求める入力と、指示の上書きをモデルへ渡さない。"""
    text = _request_text(llm_request)
    for word in _BANNED:
        if word in text:
            return _refuse("資格情報に関する質問には回答できません。天気か観光について聞いてください。")
    for word in _INJECTION:
        if word in text:
            return _refuse("その要求には応じられません。天気か観光について聞いてください。")
    return None


def redact_secrets(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """出力に鍵の形が混ざっていれば伏せる。"""
    if llm_response.content is None:
        return None
    hit = False
    for part in llm_response.content.parts or []:
        if not part.text:
            continue
        for pattern in _SECRET_PATTERNS:
            new, n = pattern.subn("***", part.text)
            if n:
                part.text = new
                hit = True
    return llm_response if hit else None
