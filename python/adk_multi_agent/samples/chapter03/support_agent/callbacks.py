"""コンテキストの制御をコールバックへ寄せる。

教材は合成関数 `compose_before_model_callbacks` を書くよう指示するが、
ADK は before/after のコールバックにリストを受け取り、順に呼んで
戻り値が真になった時点で止める。合成関数は要らない。

    before_model_callback=[rate_limit_check(10), inject_order_context()]

順序には意味がある。
上限で断る要求へコンテキストを注入しても、その注入は捨てられる。
"""

from __future__ import annotations

from collections.abc import Sequence

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

# 会員のティアと、利用者の権限を読む鍵。どちらも利用者単位なので user: を付ける。
TIER_KEY = "user:membership_tier"
ROLE_KEY = "user:role"

# 注入するアクティブな注文。セッションの間だけ持てばよい。
ACTIVE_ORDERS_KEY = "active_orders"

# 呼び出し回数の記録先。
MODEL_CALL_KEY = "model_calls"

# ツールごとに要る権限。強い順に並べて比較する。
ROLE_RANK: dict[str, int] = {"viewer": 0, "editor": 1, "admin": 2}
TOOL_PERMISSIONS: dict[str, str] = {
    "cancel_order": "editor",
    "admin_operations": "admin",
}

# ティア別の対応方針。Instruction の可変部分になる。
TIER_POLICY: dict[str, str] = {
    "free": "回答は 3 文以内にし、有料窓口の案内は行いません。",
    "standard": "回答は 5 文以内にし、代替案を 1 つ添えます。",
    "premium": "回答の長さは制限せず、代替案と次の手順を必ず添えます。優先窓口を案内できます。",
}

_BASE_INSTRUCTION = (
    "あなたはカスタマーサポートの担当です。"
    "注文の照会と取り消し、商品の検索と詳細の案内を行います。"
    "手順は list_skills と load_skill でスキルを読んでから進めます。"
    "ツールの結果に無い事実を足しません。"
)


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def build_instruction(context: ReadonlyContext) -> str:
    """ティアに応じた対応方針を含む Instruction を組み立てる。

    ReadonlyContext を受け取るので、生成の途中で State を書けない。
    同じ State からは同じ Instruction が出る。
    """
    tier = str(context.state.get(TIER_KEY, "free"))
    policy = TIER_POLICY.get(tier, TIER_POLICY["free"])
    return f"{_BASE_INSTRUCTION}\n会員区分は {tier} です。{policy}"


def rate_limit_check(
    max_calls: int = 10, message: str = "混み合っています。時間をおいて試してください。"
):
    """呼び出し回数を数え、上限を超えたらモデルを呼ばずに返す。"""

    def callback(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        used = int(callback_context.state.get(MODEL_CALL_KEY, 0)) + 1
        callback_context.state[MODEL_CALL_KEY] = used
        if used > max_calls:
            return _text_response(message)
        return None

    return callback


def inject_order_context():
    """State のアクティブな注文を、要求の末尾へ差し込む。

    末尾に置くのは、長い文脈では中間より先頭と末尾の方が参照されやすいため。
    先頭は system instruction が占める。
    """

    def callback(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        orders = callback_context.state.get(ACTIVE_ORDERS_KEY)
        if not orders:
            return None
        note = "進行中の注文: " + ", ".join(str(o) for o in orders)
        contents = list(llm_request.contents or [])
        contents.append(types.Content(role="user", parts=[types.Part(text=note)]))
        llm_request.contents = contents
        return None

    return callback


def validate_response_content(
    banned: Sequence[str] = ("パスワード", "クレジットカード番号"),
    message: str = "申し訳ありません。その内容はお伝えできません。",
):
    """禁止語を含む応答を安全な文へ差し替える。"""

    def callback(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if llm_response.content is None:
            return None
        text = "\n".join(
            part.text
            for part in llm_response.content.parts or []
            if part.text and not getattr(part, "thought", False)
        )
        if any(word in text for word in banned):
            return _text_response(message)
        return None

    return callback


def authorize_tool_access(permissions: dict[str, str] = TOOL_PERMISSIONS):
    """利用者の権限で、ツールの実行可否を決める。

    返す辞書は中身のあるものにする。空の辞書でもツールは飛ぶが、理由が残らない。
    """

    def callback(tool: BaseTool, args: dict, tool_context: ToolContext) -> dict | None:
        required = permissions.get(tool.name)
        if required is None:
            return None
        role = str(tool_context.state.get(ROLE_KEY, "viewer"))
        if ROLE_RANK.get(role, 0) >= ROLE_RANK[required]:
            return None
        return {
            "status": "error",
            "message": f"{tool.name} には {required} 以上の権限が要ります（現在 {role}）",
        }

    return callback


def trim_tool_response(field: str = "results", keep: int = 5):
    """大きな結果を上位 N 件に縮め、落とした件数を書き添える。

    件数を書かないと、モデルは全件を見たものとして答える。
    """

    def callback(
        tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: dict
    ) -> dict | None:
        items = tool_response.get(field)
        if not isinstance(items, list) or len(items) <= keep:
            return None
        trimmed = dict(tool_response)
        trimmed[field] = items[:keep]
        trimmed["note"] = f"全 {len(items)} 件のうち上位 {keep} 件"
        return trimmed

    return callback
