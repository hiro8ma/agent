# PydanticAII track

LangChain / LangGraph トラック（`../langchain/`）と同じ「エージェント抽象」を
PydanticAI で書いた比較対象。フレームワークが変わっても残る本質
（モデル / ツール / 構造化出力 / 依存注入）を一本のエージェントで示す。

## inventory エージェント

在庫の再発注を判断し、型安全な構造化結果を返す。

PydanticAI の 3 機構:

| 機構 | 実装 | LangChain 対応 |
|---|---|---|
| 構造化出力 | `output_type=ReorderDecision` | structured output / JsonOutputParser |
| ツール | `@agent.tool def get_stock(...)` | `@tool` |
| 型安全 DI | `deps_type=InventoryDeps` + `RunContext[InventoryDeps]` | `config["configurable"]` |

`ReorderDecision(item, reorder, quantity, reason)` を `output_type` に固定するため、
戻り値は LLM のテキストではなく検証済みの Pydantic インスタンスになる。

## 実行

```bash
uv sync

# オフライン（キー不要）。FunctionModel が tool → 判断 → 構造化出力を再現
uv run python bin/inventory.py --query widget-a
uv run python bin/inventory.py --query widget-b   # 在庫潤沢 → 発注なし

# 実 LLM（OPENAI_API_KEY を設定すると自動で切り替わる）
OPENAI_API_KEY=... uv run python bin/inventory.py --query gizmo-x
```

既知 SKU: `widget-a`（要発注）/ `widget-b`（潤沢）/ `gizmo-x`（しきい値ちょうど）。

## オフライン動作

キーが無い / `--fake` で PydanticAI の `FunctionModel` を使う
（公式のテスト用モデル）。`agents/inventory/fake.py` がツール呼び出しを挟み、
しきい値ロジックを通して `final_result` に構造化出力を返すので、実 LLM 抜きで
全経路を検証できる。

## 検証

```bash
uv run ruff check .
uv run mypy agents bin
uv run pytest -q
```
