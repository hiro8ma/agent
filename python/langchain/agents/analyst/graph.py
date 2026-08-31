"""データ分析エージェント。プログラマーエージェントに計画とレポートを足す。

教材の「段階的な抽象化」の第 2 段階になる。

    create_plan → （Send で並列）run_programmer → create_report
                        │
                        └── プログラマーエージェント（生成 → 実行 → レビュー）

ヘルプデスクの計画実行型と骨格が同じで、
**サブタスクの実行がコード生成・実行・レビューに置き換わる。**

同じ骨格を 2 つの業務に当てられるのが、この構成の要点になる。
教材が「ツールやプロンプトを差し替えることで多様な業務へ汎用的に適用できる」
と言うのはこのこと。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from .data import DATA_SUMMARY
from .programmer import build_graph as build_programmer
from .programmer import initial_state as programmer_state

PLAN_SYSTEM_PROMPT = """あなたはデータ分析の計画を立てます。

以下を守ってください。

- 分析タスクを 1 行ずつ、箇条書きで返す
- 説明文を書かない
- 必要最小限にする。多すぎると不要な分析が混ざる
- 同じ内容を重複して調べない
- 1 つのタスクは 1 つのコードで完結する粒度にする

例
要求: 売上の傾向を知りたい
計画:
- 月別の売上合計を出す
- 商品カテゴリ別の売上構成比を出す
"""

REPORT_SYSTEM_PROMPT = """あなたは分析結果からレポートを書きます。

以下を守ってください。

- 各分析の結果を数字で示す
- **数字だけで終わらせず、そこから言える示唆を書く**
- 不確実な情報や推測を含めない
- 分析できなかった項目は、その旨を素直に書く
- 読み手は分析の専門家ではない
"""


class AnalysisResult(TypedDict):
    index: int
    task: str
    attempts: int
    completed: bool
    code: str
    stdout: str
    result: str
    exec_log: list[dict[str, str]]


class AnalystState(TypedDict):
    request: str
    plan: list[str]
    results: Annotated[list[AnalysisResult], operator.add]
    report: str


def _invoke_text(model: BaseChatModel, system: str, user: str) -> str:
    response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = response.content
    if isinstance(content, str):
        return content.strip()
    parts = [p if isinstance(p, str) else p.get("text", "") for p in content]
    return "\n".join(str(p) for p in parts).strip()


def _build_create_plan(model: BaseChatModel) -> Any:
    def create_plan(state: AnalystState) -> dict[str, Any]:
        text = _invoke_text(
            model,
            PLAN_SYSTEM_PROMPT,
            f"要求:\n{state['request']}\n\n使えるデータ:\n{DATA_SUMMARY}",
        )
        tasks = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip("-• \t")]
        if not tasks:
            tasks = [state["request"]]
        return {"plan": tasks}


    return create_plan


def fan_out(state: AnalystState) -> list[Send]:
    """計画の要素数だけプログラマーエージェントを起動する。

    ヘルプデスクと同じ Send API の使い方になる。
    分析タスクの数は計画を作るまで決まらない。
    """
    return [
        Send("run_programmer", {"task": task, "index": i})
        for i, task in enumerate(state["plan"])
    ]


def _build_run_programmer(model: BaseChatModel) -> Any:
    """プログラマーエージェントを 1 タスクぶん動かす。

    戻す鍵は results だけにする。
    サブグラフの状態を丸ごと返すと、並列時に
    蓄積用でないフィールドが衝突する。
    """

    programmer = build_programmer(model)

    def run_programmer(state: dict[str, Any]) -> dict[str, Any]:
        task = state["task"]
        out = programmer.invoke(
            programmer_state(task, DATA_SUMMARY), {"recursion_limit": 50}
        )
        return {
            "results": [
                AnalysisResult(
                    index=state["index"],
                    task=task,
                    attempts=out["attempts"],
                    completed=out["completed"],
                    code=out["code"],
                    stdout=out["stdout"],
                    result=out["result"],
                    exec_log=list(out["exec_log"]),
                )
            ]
        }

    return run_programmer


def _build_create_report(model: BaseChatModel) -> Any:
    def create_report(state: AnalystState) -> dict[str, Any]:
        results = sorted(state["results"], key=lambda r: r["index"])

        # 渡すのはタスクと結果だけ。コードと実行ログは渡さない。
        # 中間の試行錯誤をレポートが拾うと、
        # 失敗したコードの出力まで根拠として扱われる。
        body = "\n\n".join(
            f"[{r['index']}] {r['task']}\n"
            + (r["stdout"] or r["result"] or "(結果なし)")
            for r in results
        )
        user = f"要求:\n{state['request']}\n\n分析結果:\n{body}"

        failed = [r for r in results if not r["completed"]]
        if failed:
            names = "\n".join(f"- {r['task']}" for r in failed)
            user += f"\n\n分析できなかった項目（素直に書く。推測で埋めない）:\n{names}"

        return {"report": _invoke_text(model, REPORT_SYSTEM_PROMPT, user)}

    return create_report


def build_graph(model: BaseChatModel) -> CompiledStateGraph[Any, Any, Any]:
    g: StateGraph[Any, Any, Any, Any] = StateGraph(AnalystState)
    g.add_node("create_plan", _build_create_plan(model))
    g.add_node("run_programmer", _build_run_programmer(model))
    g.add_node("create_report", _build_create_report(model))

    g.add_edge(START, "create_plan")
    g.add_conditional_edges("create_plan", fan_out, ["run_programmer"])
    g.add_edge("run_programmer", "create_report")
    g.add_edge("create_report", END)
    return g.compile()


def initial_state(request: str) -> AnalystState:
    return {"request": request, "plan": [], "results": [], "report": ""}
