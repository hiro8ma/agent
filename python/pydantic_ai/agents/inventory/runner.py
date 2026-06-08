"""在庫エージェントの実行ヘルパー。

reflection/runner.py と同じ思想で、キーがあれば実 LLM、無ければ
オフラインモデルに自動フォールバックする。
"""

from __future__ import annotations

import os

from pydantic_ai.models import Model

from .agent import InventoryDeps, ReorderDecision, _default_warehouse, build_agent
from .fake import offline_model


def _select_model(use_fake: bool) -> tuple[Model, bool]:
    """Return (model, is_fake).

    OPENAI_API_KEY があり --fake 未指定なら実 LLM、それ以外は FunctionModel。
    LLM_MODEL で OpenAI のモデル名を上書きできる（既定 gpt-4o-mini）。
    """
    if not use_fake and os.environ.get("OPENAI_API_KEY"):
        from pydantic_ai.models.openai import OpenAIChatModel

        model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        return OpenAIChatModel(model_name), False
    return offline_model(), True


def run(item: str, use_fake: bool = False) -> tuple[ReorderDecision, bool]:
    """item の再発注判断を構造化出力で返す。

    Returns (decision, is_fake)。decision は output_type で型保証された
    ReorderDecision インスタンス。
    """
    model, is_fake = _select_model(use_fake)
    agent = build_agent(model)
    deps = InventoryDeps(warehouse=_default_warehouse())
    result = agent.run_sync(f"Decide reorder for item: {item}", deps=deps)
    return result.output, is_fake
