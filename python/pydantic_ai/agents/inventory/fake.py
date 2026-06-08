"""キー無しで動かすためのオフラインモデル。

reflection/fake.py（LangChain トラック）と同じ役割。PydanticAI は
テスト用モデルを公式提供しているのでそれを使う:

- TestModel    : ツールを自動で呼び、output_type を自動充足する最小モデル
- FunctionModel: モデルの応答を関数で書ける。ここでは get_stock を呼び、
                 その結果で reorder 判断まで再現して構造化出力を返す
                 = 実 LLM 抜きで「ツール → 判断 → 構造化出力」の全経路を通す

FunctionModel を既定にして、しきい値ロジックが result type に正しく
流れることをオフラインでも検証できるようにしている。
"""

from __future__ import annotations

import json

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def _pending_item(messages: list[ModelMessage]) -> str:
    """最初のユーザ入力（'... item X ...'）から対象 SKU を取り出す。"""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                text = getattr(part, "content", None)
                if isinstance(text, str) and "item:" in text:
                    return text.split("item:", 1)[1].strip()
    return "unknown"


def _stock_from_history(messages: list[ModelMessage]) -> dict[str, int] | None:
    """get_stock の戻り値（ToolReturnPart）を履歴から拾う。"""
    for message in reversed(messages):
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolReturnPart) and part.tool_name == "get_stock":
                content = part.content
                return content if isinstance(content, dict) else None
    return None


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    item = _pending_item(messages)
    stock = _stock_from_history(messages)

    if stock is None:
        # まだ在庫を見ていない → get_stock を呼ぶ
        return ModelResponse(parts=[ToolCallPart("get_stock", {"item": item})])

    on_hand = stock["on_hand"]
    threshold = stock["threshold"]
    reorder = on_hand <= threshold
    quantity = max(0, 2 * threshold - on_hand) if reorder else 0
    reason = (
        f"on-hand {on_hand} <= threshold {threshold}, restock to ~2x threshold"
        if reorder
        else f"on-hand {on_hand} above threshold {threshold}, no action"
    )

    decision = {
        "item": item,
        "reorder": reorder,
        "quantity": quantity,
        "reason": reason,
    }
    # output_type は内部的に final_result ツール経由で受け取るためその名前を使う。
    output_tool = info.output_tools[0].name
    return ModelResponse(parts=[ToolCallPart(output_tool, json.dumps(decision))])


def offline_model() -> FunctionModel:
    return FunctionModel(_respond)
