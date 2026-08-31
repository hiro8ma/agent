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
_BROKEN = '''## 達成条件
チャネルごとの平均コンバージョン率が、3 つのチャネルすべてについて出ている。

## 実行計画
- チャネルでグループ化する
- conversion_rate の平均を取る

## コード
```python
import pandas as pd
grouped = df.groupby("channel")["conversion_rate"].mean()   # 列名が違う（正しくは channel_id）
result = grouped.to_dict()
```'''

# 2 回目。直したもの。
_FIXED = '''## 達成条件
チャネルごとの平均コンバージョン率が、3 つのチャネルすべてについて出ている。

## 実行計画
- channel_id でグループ化する
- 平均と件数を取り、平均の降順に並べる

## コード
```python
import pandas as pd
grouped = df.groupby("channel_id")["conversion_rate"].agg(["mean", "count"])
grouped = grouped.sort_values("mean", ascending=False)
print(grouped.to_string())
result = grouped["mean"].round(4).to_dict()
```'''


# 分布の分析。1 回で通る。
_DISTRIBUTION = '''## 達成条件
購入金額の平均・中央値・欠損件数が出ている。

## 実行計画
- 欠損の件数を数える
- describe で分布を出す

## コード
```python
import pandas as pd
# 購入金額の分布。欠損があるので数を先に出す。
col = df["purchase_amount"]
print(f"非欠損 {col.notna().sum()} / 全体 {len(col)}")
desc = col.describe()
print(desc.to_string())
result = {"mean": round(desc["mean"], 1), "median": round(col.median(), 1),
          "missing": int(col.isna().sum())}
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
            # 要求を 2 つの仮説に割る。1 つだと並列が見えない。
            #
            # 作業ではなく主張の形にする。
            # 「コンバージョン率を出す」は検証しても何も言えない。
            return self._reply(
                "目的: 広告費の配分を見直す\n"
                "達成条件: チャネルごとの費用対効果が比較でき、"
                "増やすべきチャネルが 1 つ以上示されている\n"
                "\n"
                "- 仮説: チャネルによってコンバージョン率に差がある\n"
                "  目的: 予算を寄せる先を決める\n"
                "  方針: channel_id ごとに conversion_rate の平均を出す\n"
                "  グラフ: 棒グラフ\n"
                "- 仮説: 購入金額は少数の高額購入に偏っている\n"
                "  目的: 平均で語ってよいかを確かめる\n"
                "  方針: purchase_amount の分布と欠損を見る\n"
                "  グラフ: ヒストグラム"
            )

        if "分析結果からレポートを書きます" in system:
            return self._reply(
                "チャネル別のコンバージョン率に大きな差は見られませんでした。\n"
                "購入金額は約 14% が欠損しており、これは購入に至らなかった行になります。\n\n"
                "示唆: 平均購入金額をチャネル比較に使うと、"
                "購入した人だけの平均になり、購入率の低いチャネルが有利に見えます。"
                "件数を併記するか、購入率と分けて見る必要があります。"
            )

        if "コードの実行結果を検査" in system:
            # 実行結果を見て判定する。コードだけでは正しさを決められない。
            failed = "エラー:\n(なし)" not in user
            empty = "result:\n(なし)" in user
            if failed or empty:
                return self._reply(
                    "COMPLETED: false\n"
                    "OBSERVATION: 実行に失敗しているか結果が空です。"
                    "列名と添字の扱いを見直してください。"
                )
            return self._reply(
                "COMPLETED: true\nOBSERVATION: 達成条件を満たしています。"
            )

        # タスクの内容でコードを切り替える。
        # 分布の分析は 1 回で通し、平均の分析だけ 1 回失敗させる。
        # 全部失敗させると、成功だけの経路が試されない。
        if "購入金額" in user or "分布" in user:
            return self._reply(_DISTRIBUTION)

        n = self._calls.get("code", 0)
        self._calls["code"] = n + 1
        return self._reply(_BROKEN if n == 0 else _FIXED)

    def _reply(self, text: str) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
