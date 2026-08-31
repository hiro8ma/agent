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

from .dataset import describe_dataframe
from .plan import Task, parse_plan
from .programmer import build_graph as build_programmer
from .programmer import initial_state as programmer_state

PLAN_SYSTEM_PROMPT = """あなたはデータ分析の計画を立てます。

レポートは 60 分の会議で使われます。
方針を決めるための叩き台になるため、作業の一覧ではなく
**検証できる仮説**を立ててください。

要求には曖昧な部分があります。意図を推測して補ってください。

次の形式で返してください。説明文は書かないでください。

目的: 要求から読み取れる問い合わせの目的
達成条件: 何が示されればこの要求に答えたと言えるか

- 仮説: 検証できる形の主張。「〜を出す」ではなく「〜は〜より高い」
  目的: この仮説を確かめる理由
  方針: どの列をどう集計するか
  グラフ: 棒グラフ / 折れ線グラフ / ヒストグラム / 散布図 のいずれか

仮説は必要最小限にしてください。重複させないでください。
1 つの仮説は 1 本のコードで検証できる粒度にしてください。
"""

REPORT_SYSTEM_PROMPT = """あなたは分析結果からレポートを書きます。
レポートは 60 分の会議で方針を決めるための資料になります。

以下を守ってください。

- 仮説ごとにセクションを分け、検証結果と示唆を書く
- 各分析の結果を数字で示す
- **数字だけで終わらせず、そこから言える示唆を書く**
- 不確実な情報や推測を含めない
- 分析できなかった項目は、その旨を素直に書く
- 読み手は分析の専門家ではない
- **最後に達成条件を振り返り、足りない情報を次にやることとして書く**

次の構成にしてください。

# データ分析レポート
## 分析の目的
## 分析結果詳細
### 仮説 1: ...
（検証結果と示唆）
## まとめと考察
## ネクストアクション
"""


class AnalysisResult(TypedDict):
    index: int
    task: str
    purpose: str
    chart_type: str
    observation: str
    attempts: int
    completed: bool
    code: str
    stdout: str
    result: str
    exec_log: list[dict[str, str]]


class AnalystState(TypedDict):
    request: str
    # plan は仮説の一覧。作業ではなく主張を並べる。
    plan: list[dict[str, str]]
    purpose: str
    # achievement は計画全体の達成条件。レポートの基準になる。
    #
    # コード 1 本ごとの達成条件（Program.achievement_condition）とは別物。
    # 個々のコードが通っても問い合わせに答えていない、
    # という結果を防ぐには両方が要る。
    achievement: str
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
            f"要求:\n{state['request']}\n\n使えるデータ:\n{describe_dataframe()}",
        )
        parsed = parse_plan(text)
        tasks = [t.model_dump() for t in parsed.tasks]
        if not tasks:
            # 仮説が読み取れなくても分析は進める。
            # 要求そのものを 1 件の仮説として扱う。
            tasks = [Task(hypothesis=state["request"]).model_dump()]
        return {
            "plan": tasks,
            "purpose": parsed.purpose,
            "achievement": parsed.achievement,
        }


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
        # 仮説と方針とグラフ種別をまとめてコード生成へ渡す。
        # グラフ種別を渡すのは自由度を下げるため。
        # 「いい感じに可視化して」だと毎回違うものが出て比較できない。
        request = task["hypothesis"]
        if task.get("description"):
            request += f"\n分析方針: {task['description']}"
        if task.get("chart_type"):
            request += f"\nグラフ: {task['chart_type']} で描く"

        out = programmer.invoke(
            programmer_state(request, describe_dataframe()), {"recursion_limit": 50}
        )
        return {
            "results": [
                AnalysisResult(
                    index=state["index"],
                    task=task["hypothesis"],
                    purpose=task.get("purpose", ""),
                    chart_type=task.get("chart_type", ""),
                    observation=out.get("observation", ""),
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
            f"仮説 {r['index'] + 1}: {r['task']}\n"
            f"検証目的: {r['purpose'] or '(なし)'}\n"
            f"結果:\n{r['stdout'] or r['result'] or '(結果なし)'}\n"
            f"レビューの所見: {r['observation'] or '(なし)'}"
            for r in results
        )
        user = (
            f"要求:\n{state['request']}\n\n"
            f"分析の目的:\n{state.get('purpose') or '(なし)'}\n\n"
            f"達成条件（ネクストアクションはこれを基準に書く）:\n"
            f"{state.get('achievement') or '(なし)'}\n\n"
            f"分析結果:\n{body}"
        )

        failed = [r for r in results if not r["completed"]]
        if failed:
            names = "\n".join(f"- {r['task']}" for r in failed)
            user += f"\n\n分析できなかった項目（素直に書く。推測で埋めない）:\n{names}"

        return {"report": _invoke_text(model, REPORT_SYSTEM_PROMPT, user)}

    return create_report


def build_graph(
    model: BaseChatModel, usage: Any = None
) -> CompiledStateGraph[Any, Any, Any]:
    """グラフを組む。usage を渡すと段ごとの呼び出し回数と費用を数える。

    数える場所をモデルの手前に置く。
    各ノードに数える処理を書くと、書き忘れた箇所が静かに漏れる。
    """

    def wrap(node: str) -> BaseChatModel:
        if usage is None:
            return model
        from .usage import MeteredModel

        return MeteredModel(inner=model, usage=usage, node=node)

    g: StateGraph[Any, Any, Any, Any] = StateGraph(AnalystState)
    g.add_node("create_plan", _build_create_plan(wrap("create_plan")))
    g.add_node("run_programmer", _build_run_programmer(wrap("programmer")))
    g.add_node("create_report", _build_create_report(wrap("create_report")))

    g.add_edge(START, "create_plan")
    g.add_conditional_edges("create_plan", fan_out, ["run_programmer"])
    g.add_edge("run_programmer", "create_report")
    g.add_edge("create_report", END)
    return g.compile()


def initial_state(request: str) -> AnalystState:
    return {
        "request": request,
        "plan": [],
        "purpose": "",
        "achievement": "",
        "results": [],
        "report": "",
    }
