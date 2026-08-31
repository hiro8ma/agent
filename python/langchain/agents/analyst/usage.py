"""LLM の使用量と費用を数える。

請求の反映は最大 24 時間遅れる。
予算アラートも「通知するだけで止めない」仕組みになる。
**実行中に止めるには、自分で数えるしかない。**

自己修正は上限まで回れば呼び出しが 3 倍になる。
試行回数だけを見ていると、その費用が見えない。
「2 回で通った」と「3 回で諦めた」は、結果だけでなく費用でも違う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

# 100 万トークンあたりの単価。公式の料金表から取る。
# 単価は変わるので、値そのものではなく出典を残す。
COST_PER_MTOK: dict[str, dict[str, float]] = {
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gemini-3.5-flash": {"input": 0.30, "output": 2.50},
}


@dataclass
class Usage:
    """1 回の実行を通した使用量。

    呼び出しごとではなく実行全体で持つ。
    「この分析にいくらかかったか」が知りたい単位はそちらになる。
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    # per_node は段ごとの内訳。どこが高いかを見る。
    per_node: dict[str, int] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float | None:
        """概算費用。単価が分からないモデルでは None を返す。

        0 を返してはいけない。「無料」と「測っていない」を
        同じ数字にすると、費用ゼロだと誤読される。
        """

        rate = COST_PER_MTOK.get(self.model)
        if rate is None:
            return None
        return (
            self.input_tokens * rate["input"] + self.output_tokens * rate["output"]
        ) / 1_000_000

    def record(self, node: str, response: Any) -> None:
        self.calls += 1
        self.per_node[node] = self.per_node.get(node, 0) + 1
        meta = getattr(response, "usage_metadata", None) or {}
        self.input_tokens += int(meta.get("input_tokens", 0) or 0)
        self.output_tokens += int(meta.get("output_tokens", 0) or 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "perNode": dict(self.per_node),
            "costUsd": self.cost_usd,
            "model": self.model or "(不明)",
        }


class MeteredModel(BaseChatModel):
    """呼び出しを数えるためのラッパー。

    モデルそのものを差し替えず、呼び出しの前後で数える。
    数える処理を各ノードに書くと、書き忘れた箇所が静かに漏れる。
    """

    inner: BaseChatModel
    usage: Usage
    node: str = "unknown"

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return f"metered-{self.inner._llm_type}"

    def _generate(self, messages: list[BaseMessage], stop: Any = None,
                  run_manager: Any = None, **kwargs: Any) -> Any:
        result = self.inner._generate(messages, stop, run_manager, **kwargs)
        gen = result.generations[0] if result.generations else None
        self.usage.record(self.node, getattr(gen, "message", None))
        return result

    def bind(self, **kwargs: Any) -> Any:
        return self.inner.bind(**kwargs)
