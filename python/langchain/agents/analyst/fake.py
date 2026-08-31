"""API キー無しで動かすための代替モデル。

生成する「コード」は固定だが、**わざと 1 回目を失敗させる**。
失敗しない代替モデルでは、やり直しの経路が一度も通らない。
ヘルプデスク側で、空振りを NG にしない代替モデルのせいで
未解決の経路が試せていなかったのと同じ問題になる。
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

# 1 回目。存在しない列を参照して失敗する。
_BROKEN = '''```python
# 平均評価値をジャンル別に出す
from collections import defaultdict
by_genre = defaultdict(list)
for r in ratings:
    m = movies[r["movie_id"]]          # 添字と id を取り違えている
    for g in m["genre"]:               # 列名が違う（正しくは genres）
        by_genre[g].append(r["rating"])
result = {g: sum(v)/len(v) for g, v in by_genre.items()}
```'''

# 2 回目。直したもの。
_FIXED = '''```python
from collections import defaultdict
by_id = {m["movie_id"]: m for m in movies}
by_genre = defaultdict(list)
for r in ratings:
    m = by_id.get(r["movie_id"])
    if not m:
        continue
    for g in m["genres"]:
        by_genre[g].append(r["rating"])
result = {g: round(sum(v)/len(v), 3) for g, v in sorted(by_genre.items())}
print("ジャンル数:", len(result))
for g, v in result.items():
    print(f"  {g:8s} {v}")
```'''


# 分布の分析。1 回で通る。
_DISTRIBUTION = '''```python
from collections import Counter
import statistics

counts = Counter(r["rating"] for r in ratings)
values = [r["rating"] for r in ratings]
mean = statistics.mean(values)
median = statistics.median(values)
print(f"平均 {mean:.3f} / 中央値 {median:.1f}")
for v in sorted(counts):
    print(f"  {v}: {counts[v]:4d}")
result = {"mean": round(mean, 3), "median": median,
          "counts": {str(k): v for k, v in sorted(counts.items())}}
```'''


class FakeAnalystModel(BaseChatModel):
    """ネットワーク不要。1 回目に失敗するコードを返し、2 回目で直す。"""

    _calls: dict[str, int] = PrivateAttr(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "fake-analyst"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = messages[0].content if messages else ""
        system = system if isinstance(system, str) else ""
        user = messages[-1].content if messages else ""
        user = user if isinstance(user, str) else ""

        if "データ分析の計画を立てます" in system:
            # 要求を 2 つの分析タスクに割る。1 つだと並列が見えない。
            return self._reply(
                "- ジャンルごとの平均評価値を出す\n"
                "- 評価値の分布を出す"
            )

        if "分析結果からレポートを書きます" in system:
            return self._reply(
                "分析の結果、ジャンル間で平均評価値に大きな差はありませんでした"
                "（いずれも 3.9 前後）。\n"
                "評価値そのものは高い側に偏っており、中央値が平均を上回ります。\n\n"
                "示唆: ジャンルで出し分ける根拠は弱く、"
                "評価値をそのまま順位付けに使うと高評価側に潰れます。"
            )

        if "コードの実行結果を検査" in system:
            # 実行結果を見て判定する。コードだけでは正しさを決められない。
            failed = "エラー:\n(なし)" not in user
            empty = "result:\n(なし)" in user
            if failed or empty:
                return self._reply(
                    "VERDICT: RETRY\n"
                    "実行に失敗しているか結果が空です。列名と添字の扱いを見直してください。"
                )
            return self._reply("VERDICT: PASS\n")

        # タスクの内容でコードを切り替える。
        # 分布の分析は 1 回で通し、平均の分析だけ 1 回失敗させる。
        # 全部失敗させると、成功だけの経路が試されない。
        if "分布" in user:
            return self._reply(_DISTRIBUTION)

        n = self._calls.get("code", 0)
        self._calls["code"] = n + 1
        return self._reply(_BROKEN if n == 0 else _FIXED)

    def _reply(self, text: str) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
