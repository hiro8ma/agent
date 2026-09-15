"""コールバックを並べて使う。

ADK の before/after 系コールバックは、単体でもリストでも渡せる。
リストを渡すと ADK が順に呼び、戻り値が真になった時点で止める。
合成する関数を自前で書く必要はない。

    before_model_callback=[log_requests(log), limit_model_calls(3), inject_note(...)]

打ち切りとツール実行では、判定している条件が違う。

    鎖を止めるか      `if function_response:`        真のときだけ止まる
    ツールを呼ぶか    `if function_response is None:`  None のときだけ呼ぶ

そのため before_tool で空の dict を返すと、鎖は進むのにツールは飛ぶ。
後ろのコールバックが真の値を返せば、そちらが結果になる。実測で確かめた。
返すなら中身のある値にし、通したいときは None を返す。

関心事ごとに関数を分けておくと、順序の入れ替えと単体での検査ができる。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.sessions.state import State
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

# 呼び出し回数の記録先。セッションの間だけ持てばよい。
MODEL_CALL_KEY = "model_calls"

# 権限の判定に使う鍵。利用者単位なので user: を付ける。
ROLE_KEY = State.USER_PREFIX + "role"


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def request_text(request: LlmRequest) -> str:
    """要求に含まれる本文。thought は除く。

    parts[0] だけを見るとツール呼び出しや thought の回で取りこぼす。
    """
    out = []
    for content in request.contents or []:
        for part in content.parts or []:
            if part.text and not getattr(part, "thought", False):
                out.append(part.text)
    return "\n".join(out)


def response_text(response: LlmResponse) -> str:
    """応答の本文。thought は除く。"""
    if response.content is None:
        return ""
    return "\n".join(
        part.text
        for part in response.content.parts or []
        if part.text and not getattr(part, "thought", False)
    )


def log_requests(sink: list[str]) -> Callable[[CallbackContext, LlmRequest], None]:
    """記録だけして通す。None を返すので後続も走る。"""

    def callback(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        sink.append(f"{callback_context.agent_name}:{len(llm_request.contents or [])}")

    return callback


def limit_model_calls(
    max_calls: int,
    message: str = "呼び出しの上限に達しました。時間をおいて試してください。",
):
    """1 セッションでのモデル呼び出しを数え、超えたら呼ばずに返す。

    返すのは LlmResponse なので、ここで鎖が止まり、モデルも呼ばれない。
    """

    def callback(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        used = int(callback_context.state.get(MODEL_CALL_KEY, 0)) + 1
        callback_context.state[MODEL_CALL_KEY] = used
        if used > max_calls:
            return _text_response(message)
        return None

    return callback


def inject_note(note: str):
    """補足を末尾に足す。

    末尾に置くのは、長い文脈では中間より先頭と末尾の方が参照されやすいため。
    先頭は system instruction が占めるので、差し込みは末尾にする。
    """

    def callback(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        contents = list(llm_request.contents or [])
        contents.append(types.Content(role="user", parts=[types.Part(text=note)]))
        llm_request.contents = contents

    return callback


def redact_response(words: Sequence[str], message: str):
    """禁止語を含む応答を差し替える。

    LlmResponse を返すと、その応答に差し替わって後続のコールバックは走らない。
    """

    def callback(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        text = response_text(llm_response)
        if any(word in text for word in words):
            return _text_response(message)
        return None

    return callback


def authorize_tools(
    write_tools: Sequence[str], allowed_roles: Sequence[str] = ("admin",)
):
    """書き込み系のツールを権限で止める。

    dict を返すとツールは実行されず、その値がツールの結果になる。
    通すときは None を返す。空の dict はツールを止めるが鎖は止めないので、理由が消える。
    """
    write = set(write_tools)
    allowed = set(allowed_roles)

    def callback(tool: BaseTool, args: dict, tool_context: ToolContext) -> dict | None:
        if tool.name not in write:
            return None
        role = tool_context.state.get(ROLE_KEY, "viewer")
        if role in allowed:
            return None
        return {
            "status": "error",
            "message": f"{tool.name} には {'/'.join(sorted(allowed))} の権限が要る（現在 {role}）",
        }

    return callback


def shrink_tool_result(field: str, keep: int, tools: Sequence[str] | None = None):
    """大きなツール結果を上位 N 件に縮める。

    そのままモデルへ渡すとコンテキストを食う。
    落とした件数を書き添えないと、モデルは全件見たものとして答える。
    """
    target = set(tools) if tools else None

    def callback(
        tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: dict
    ) -> dict | None:
        if target is not None and tool.name not in target:
            return None
        items = tool_response.get(field)
        if not isinstance(items, list) or len(items) <= keep:
            return None
        shrunk = dict(tool_response)
        shrunk[field] = items[:keep]
        shrunk["note"] = f"全 {len(items)} 件のうち上位 {keep} 件"
        return shrunk

    return callback
