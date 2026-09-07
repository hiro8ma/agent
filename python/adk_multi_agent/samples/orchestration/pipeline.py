"""Template Workflow の入れ子と、その 2 つの落とし穴。

SequentialAgent と ParallelAgent と LoopAgent は v2.2.0 で 3 つとも
非推奨になっている。実行すると警告が出る。

    <名前> is deprecated and will be removed in future versions.
    Please use Workflow instead.

新規に書くなら Workflow を使う。ここに残すのは、既存コードを読む側で
必要になるためと、落とし穴 2 つが Workflow でも同じ形で出るため。

落とし穴はどちらもエラーを出さない。

    max_iterations 未設定    止まらない
    output_key の衝突        後から書いた方だけが残る

この 2 つを検査で固定する。
"""

from __future__ import annotations

from google.adk import Agent
from google.adk.agents import LoopAgent, ParallelAgent, SequentialAgent

# 反復の上限。LoopAgent の既定は None で、None は「止まらない」になる。
MAX_ROUNDS = 3


def build_loop(generator: Agent, reviewer: Agent) -> LoopAgent:
    """生成と評価を繰り返す。上限を必ず渡す。"""
    return LoopAgent(
        name="refinement_loop",
        sub_agents=[generator, reviewer],
        max_iterations=MAX_ROUNDS,
    )


def build_nested(
    searches: list[Agent], planner: Agent, reporter: Agent
) -> SequentialAgent:
    """検索を並列にし、その後を順次にする。

    並列の子は同じ state を共有する。output_key が重なると、
    後から書いた方だけが残り、他は失われる。呼び出しの費用は払う。
    """
    keys = [a.output_key for a in searches if a.output_key]
    if len(set(keys)) != len(keys):
        raise ValueError(f"output_key が重複している: {keys}")

    return SequentialAgent(
        name="travel_pipeline",
        sub_agents=[
            ParallelAgent(name="research_phase", sub_agents=list(searches)),
            planner,
            reporter,
        ],
    )
