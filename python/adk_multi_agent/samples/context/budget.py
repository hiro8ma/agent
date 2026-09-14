"""コンテキスト予算と、汚染への 2 つの手当て。

コンテキストウィンドウは有限で、詰めるほど遅く高くなる。
さらに、長いほど中間に置いた情報の参照率が落ちる（Lost in the Middle）。
そこで、1 回の呼び出しで各カテゴリへ何トークン配るかを先に決める。

    ContextBudget      割り当てを決める
    enforce_budget     before_model_callback として実際に削る

削る対象は 2 つ。

    蓄積型   古いツール結果が新しい結果の解釈を歪める。直近 N 件だけ残す
    溢れ     履歴が予算を超える。古い順に落とし、最後の利用者発話は必ず残す

トークン数は概算で数える。正確な値はモデル側の count_tokens でしか出ないが、
削る判断は概算で足りるうえ、呼び出しを増やさずに済む。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.sessions.state import State
from google.genai import types

# 概算の換算。日本語は 1 トークンあたりの文字数が英語より少ないので、短めに見積もる。
CHARS_PER_TOKEN = 2.0

# 削った結果を書く鍵。temp: はセッションに残らないので、監視用の値の置き場に向く。
TRIM_REPORT_KEY = State.TEMP_PREFIX + "context_trim"


def rough_token_estimate(text: str) -> int:
    """文字数からの概算。正確な値ではない。"""
    return int(len(text) / CHARS_PER_TOKEN + 0.5)


def content_tokens(content: types.Content) -> int:
    """1 つの Content の概算トークン数。"""
    total = 0
    for part in content.parts or []:
        if part.text:
            total += rough_token_estimate(part.text)
        if part.function_call is not None:
            total += rough_token_estimate(str(part.function_call.args or {}))
        if part.function_response is not None:
            total += rough_token_estimate(str(part.function_response.response or {}))
    return total


@dataclass(frozen=True)
class ContextBudget:
    """1 回の呼び出しでの割り当て。

    比率は合計 1.0 にする。合計が合わないまま運用すると、
    どこかのカテゴリが黙って溢れる。
    """

    total_tokens: int = 128_000
    shares: dict[str, float] = field(
        default_factory=lambda: {
            "system_instruction": 0.10,
            "user_info": 0.05,
            "history": 0.30,
            "tool_results": 0.25,
            "dynamic": 0.15,
            "output": 0.15,
        }
    )

    def __post_init__(self) -> None:
        if self.total_tokens <= 0:
            raise ValueError("total_tokens は正の値にする")
        total = sum(self.shares.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"割り当ての合計が 1.0 でない: {total}")

    def limit(self, category: str) -> int:
        """カテゴリごとのトークン上限。"""
        if category not in self.shares:
            raise KeyError(f"未定義のカテゴリ: {category}")
        return int(self.total_tokens * self.shares[category])

    @property
    def input_limit(self) -> int:
        """出力用バッファを除いた入力側の上限。"""
        return self.total_tokens - self.limit("output")


def drop_stale_tool_results(
    contents: list[types.Content], keep: int
) -> list[types.Content]:
    """ツール結果を直近 keep 件だけ残す。

    古い結果を残すと、新しい結果と並んだときにどちらが現在の値か決められない。
    在庫ありと在庫なしが同居する類の矛盾はここで生まれる。
    """
    if keep < 0:
        raise ValueError("keep は 0 以上にする")
    indexes = [
        i
        for i, c in enumerate(contents)
        if any(p.function_response is not None for p in c.parts or [])
    ]
    drop = set(indexes[: max(0, len(indexes) - keep)])
    return [c for i, c in enumerate(contents) if i not in drop]


def trim_to_limit(contents: list[types.Content], limit: int) -> list[types.Content]:
    """上限に収まるまで古い順に落とす。最後の利用者発話は必ず残す。

    最後の発話まで落とすと、モデルは何を聞かれたのか分からなくなる。
    """
    if not contents:
        return contents
    keep_last = contents[-1]
    budgeted = [keep_last]
    used = content_tokens(keep_last)
    for content in reversed(contents[:-1]):
        cost = content_tokens(content)
        if used + cost > limit:
            break
        budgeted.append(content)
        used += cost
    budgeted.reverse()
    return budgeted


def enforce_budget(budget: ContextBudget, keep_tool_results: int = 2):
    """予算を守らせる before_model_callback を作る。

    返り値が None なら、そのままモデルへ進む。
    ここでは要求を書き換えるだけで、応答は作らない。
    """

    def callback(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        original = list(llm_request.contents or [])
        if not original:
            return None

        kept = drop_stale_tool_results(original, keep_tool_results)
        kept = trim_to_limit(kept, budget.input_limit)
        llm_request.contents = kept

        before = sum(content_tokens(c) for c in original)
        after = sum(content_tokens(c) for c in kept)
        callback_context.state[TRIM_REPORT_KEY] = {
            "contents_before": len(original),
            "contents_after": len(kept),
            "tokens_before": before,
            "tokens_after": after,
            "input_limit": budget.input_limit,
        }
        return None

    return callback
