"""カスタマーサポートのツール群。

docstring と型注釈はモデルへの説明になる。
引数の例まで書かないと、それらしい値を作られる。

副作用のある処理（記録の保存）はツールに置く。
コールバックへ置くと、どの操作で何が残ったのかが追えなくなる。
"""

from __future__ import annotations

from google.adk.tools import ToolContext
from google.genai import types

# 固定データ。本番では注文管理の API を呼ぶ。
_ORDERS: dict[str, dict] = {
    "ORD-12345": {"status": "preparing", "shipped": False, "total_yen": 12800},
    "ORD-99999": {"status": "shipped", "shipped": True, "total_yen": 4200},
}

_PRODUCTS: list[dict] = [
    {
        "product_id": f"P-{i:03d}",
        "name": f"商品{i}",
        "category": "apparel",
        "price_yen": 1000 + i * 100,
    }
    for i in range(1, 11)
]


def get_order_status(order_id: str) -> dict:
    """注文の現在の状態を取得する。

    Args:
        order_id: 注文 ID。"ORD-" で始まる英数字。例: ORD-12345

    Returns:
        order_id と status と shipped を含む辞書。
        見つからないときは error を含む辞書を返す
    """
    order = _ORDERS.get(order_id)
    if order is None:
        return {"error": f"{order_id} は見つかりません"}
    return {"order_id": order_id, **order}


async def cancel_order(order_id: str, reason: str, tool_context: ToolContext) -> dict:
    """注文を取り消し、取り消し記録を成果物として残す。

    Args:
        order_id: 注文 ID。例: ORD-12345
        reason: 取り消しの理由。利用者の言葉をそのまま入れる。例: サイズが合わない

    Returns:
        order_id と status と artifact_version を含む辞書。
        発送済みや未登録のときは error を含む辞書を返す
    """
    order = _ORDERS.get(order_id)
    if order is None:
        return {"error": f"{order_id} は見つかりません"}
    if order["shipped"]:
        return {
            "error": f"{order_id} は発送済みのため取り消せません。返品の案内へ切り替えてください"
        }

    record = types.Part.from_text(
        text=f"キャンセル記録\n注文ID: {order_id}\n理由: {reason}\n"
    )
    version = await tool_context.save_artifact(f"cancel_{order_id}.txt", record)
    return {"order_id": order_id, "status": "cancelled", "artifact_version": version}


def search_products(query: str, category: str) -> dict:
    """商品を検索する。

    Args:
        query: 検索語。例: シャツ
        category: 絞り込むカテゴリ。空文字なら絞り込まない。例: apparel

    Returns:
        results に商品の一覧を含む辞書
    """
    items = _PRODUCTS
    if category:
        items = [p for p in items if p["category"] == category]
    return {"query": query, "results": items}


def get_product_details(product_id: str) -> dict:
    """商品の詳細を取得する。

    Args:
        product_id: 商品 ID。"P-" で始まる。例: P-001

    Returns:
        商品の詳細。見つからないときは error を含む辞書を返す
    """
    for product in _PRODUCTS:
        if product["product_id"] == product_id:
            return {**product, "in_stock": True}
    return {"error": f"{product_id} は見つかりません"}
