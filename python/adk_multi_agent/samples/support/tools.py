"""サポート担当のツール。

search_products は情報源を決まった順に引き、足りた時点で止める。
順番を Instruction で頼むと、モデルが守るかどうかに依存する。
しかも Gemini では VertexAiRagRetrieval の名前がモデルに届かないので、名前で順番を指示できない。
"""

from __future__ import annotations

from google.adk.tools import ToolContext

from samples.state.keys import StateKeys, get_max_results

TEMP_LAST_TOOL_RESULT = "temp:last_tool_result"

# 情報源と文書。実運用では RAG Engine のコーパスや検索サービスに置き換える。
SOURCES: list[tuple[str, list[dict[str, str]]]] = [
    (
        "product_docs",
        [
            {"title": "ワイヤレスイヤホン X1", "body": "連続再生 8 時間。防水 IPX4。"},
            {
                "title": "ワイヤレスイヤホン X2",
                "body": "連続再生 12 時間。ノイズキャンセリング対応。",
            },
        ],
    ),
    (
        "faq_docs",
        [
            {"title": "返品の期限", "body": "返品は商品到着から 14 日以内。"},
            {"title": "保証", "body": "保証期間は購入から 1 年。"},
        ],
    ),
]

ORDERS = {
    "A1": {"status": "shipped", "carrier": "yamato"},
    "A2": {"status": "preparing", "carrier": None},
}


def _matches(query: str, doc: dict[str, str]) -> bool:
    text = doc["title"] + doc["body"]
    return any(term in text for term in query.split())


def search_products(query: str, tool_context: ToolContext) -> dict:
    """商品の仕様と、返品や保証などのよくある質問を検索する。

    Args:
        query: 検索語。複数の語は空白で区切る。
    """
    state = tool_context.state
    state[StateKeys.TEMP_SEARCH_COUNT] = state.get(StateKeys.TEMP_SEARCH_COUNT, 0) + 1
    limit = get_max_results(state)

    searched: list[str] = []
    for source, docs in SOURCES:
        searched.append(source)
        hits = [d for d in docs if _matches(query, d)][:limit]
        if hits:
            result = {"source": source, "searched": searched, "results": hits}
            state[TEMP_LAST_TOOL_RESULT] = result
            return result
    result = {"source": None, "searched": searched, "results": []}
    state[TEMP_LAST_TOOL_RESULT] = result
    return result


def get_order_status(order_id: str, tool_context: ToolContext) -> dict:
    """注文の配送状況を返す。

    Args:
        order_id: 注文番号。
    """
    order = ORDERS.get(order_id)
    result = {"order_id": order_id, "found": order is not None, **(order or {})}
    tool_context.state[TEMP_LAST_TOOL_RESULT] = result
    return result
