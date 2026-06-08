"""型安全な PydanticAI 在庫管理エージェント

LangChain トラックの reflection / helpdesk と同じ「エージェント抽象」を
PydanticAI で書いた比較対象。PydanticAI の 3 つの型安全機構を 1 本で示す:

- output_type   : 戻り値を Pydantic モデルに固定（構造化出力 = 型安全な result）
- @agent.tool   : 実行時に LLM が呼ぶツール（在庫データ取得を fake で提供）
- deps_type     : RunContext 経由で実行時依存を注入（型安全 DI）

LangChain 対応:
  Agent(model, system_prompt, output_type)  ≈ create_agent + structured output
  @agent.tool / RunContext[Deps]            ≈ @tool + config["configurable"] DI
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

SYSTEM_PROMPT = (
    "You are an inventory planner. Given a stock keeping unit, decide whether to "
    "reorder. Always call the get_stock tool to read current stock and the reorder "
    "threshold before deciding. Reorder when on-hand stock is at or below the "
    "threshold. Choose a reorder quantity that brings stock back to roughly twice "
    "the threshold. Keep the reason to one sentence."
)


class ReorderDecision(BaseModel):
    """エージェントの構造化出力（型安全な result type）。"""

    item: str = Field(..., description="The SKU the decision is about.")
    reorder: bool = Field(..., description="True when a purchase order should be raised.")
    quantity: int = Field(..., ge=0, description="Units to reorder; 0 when no reorder.")
    reason: str = Field(..., description="One-sentence justification.")


@dataclass
class StockRecord:
    on_hand: int
    threshold: int


@dataclass
class InventoryDeps:
    """deps_type で注入する実行時依存。

    本番なら DB / 外部 API ハンドルを持たせる箇所。ここでは fake の
    在庫テーブルを差し替え可能な形で渡し、DI の型安全さを示す。
    """

    warehouse: dict[str, StockRecord] = field(default_factory=dict)


def _default_warehouse() -> dict[str, StockRecord]:
    return {
        "widget-a": StockRecord(on_hand=4, threshold=10),
        "widget-b": StockRecord(on_hand=120, threshold=30),
        "gizmo-x": StockRecord(on_hand=10, threshold=10),
    }


def build_agent(model: Model) -> Agent[InventoryDeps, ReorderDecision]:
    """エージェントを組み立てて返す。

    model は Model インスタンス（実 LLM は OpenAIChatModel、オフラインは
    FunctionModel / TestModel）を受ける。生成は runner 側で行う。
    """
    agent: Agent[InventoryDeps, ReorderDecision] = Agent(
        model,
        deps_type=InventoryDeps,
        output_type=ReorderDecision,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.tool
    def get_stock(ctx: RunContext[InventoryDeps], item: str) -> dict[str, int]:
        """Return current on-hand stock and reorder threshold for an SKU."""
        record = ctx.deps.warehouse.get(item)
        if record is None:
            raise ModelRetry(f"unknown item {item!r}; valid items: {sorted(ctx.deps.warehouse)}")
        return {"on_hand": record.on_hand, "threshold": record.threshold}

    return agent
