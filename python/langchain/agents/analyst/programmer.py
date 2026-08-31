"""プログラマーエージェント。コード生成・実行・レビューを繰り返す。

教材の三分（計画・行動・知覚）をそのまま置く。

    generate_code（計画） → execute（行動） → review（知覚）
          ↑                                        │
          └────────── 直らないなら上限まで ─────────┘

ヘルプデスクの計画実行型と骨格は同じで、**行動が検索ではなくコード実行**になる。
違いはそこから出てくる。

  検索は失敗しても何も壊れない。コード実行は環境を壊しうる
  検索の失敗は「見つからない」の 1 種類。実行の失敗は例外・タイムアウト・
  意図と違う結果、と種類が増える
  検索結果は読めば分かる。実行結果は「正しいか」を別途判定する必要がある
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .sandbox import run_code

MAX_ATTEMPTS = 3

CODE_SYSTEM_PROMPT = """あなたはデータ分析のコードを書きます。

以下を必ず守ってください。

- Python のコードだけを ```python のコードブロックで返す
- 説明文を書かない
- 与えられた変数だけを使う。ファイルやネットワークにアクセスしない
- ライブラリは自分で import する（pandas は使える）
- 最後に `result` という変数へ答えを入れる
- print で途中経過を出す

前回のコードが失敗している場合は、エラーの内容を読んで直したコードを返してください。
同じコードを繰り返さないでください。
"""

REVIEW_SYSTEM_PROMPT = """あなたはコードの実行結果を検査します。

2 行で答えてください。

- 1 行目: タスクに正しく答えられていれば `VERDICT: PASS`、そうでなければ `VERDICT: RETRY`
- 2 行目: RETRY のとき、何が問題でどう直すかを 1 文で

実行が失敗した場合、結果が空の場合、タスクと無関係な値の場合は RETRY にしてください。
"""


class ProgrammerState(TypedDict):
    """状態。

    tool_log にあたるものが exec_log になる。
    どのコードを走らせて何が返ったかを残さないと、
    レビューが「失敗したこと」しか分からない。
    """

    task: str
    # data_summary は解析対象の説明。コードを書くのに要る。
    data_summary: str
    code: str
    attempts: int
    completed: bool
    exec_log: Annotated[list[dict[str, str]], operator.add]
    advice: Annotated[list[str], operator.add]
    stdout: str
    result: str
    error: str


def _invoke_text(model: BaseChatModel, system: str, user: str) -> str:
    response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = response.content
    if isinstance(content, str):
        return content.strip()
    parts = [p if isinstance(p, str) else p.get("text", "") for p in content]
    return "\n".join(str(p) for p in parts).strip()


def _extract_code(text: str) -> str:
    """返答からコードブロックを取り出す。

    説明文が混ざったまま実行すると必ず SyntaxError になる。
    「コードだけ返せ」と指示しても混ざるので、受け取る側で外す。
    """

    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return blocks[0].strip() if blocks else text.strip()


def _build_generate(model: BaseChatModel) -> Any:
    def generate_code(state: ProgrammerState) -> dict[str, Any]:
        user = (
            f"タスク:\n{state['task']}\n\n"
            f"使えるデータ:\n{state['data_summary']}"
        )
        # 前回の失敗を渡す。渡さないと同じコードを書き直す。
        if state["attempts"] > 0:
            user += (
                f"\n\n前回のコード:\n```python\n{state['code']}\n```"
                f"\n\n実行結果:\n{state['error'] or state['stdout']}"
            )
            if state["advice"]:
                joined = "\n".join(f"- {a}" for a in state["advice"])
                user += f"\n\n指摘:\n{joined}"
        text = _invoke_text(model, CODE_SYSTEM_PROMPT, user)
        return {"code": _extract_code(text), "attempts": state["attempts"] + 1}

    return generate_code


def _execute(state: ProgrammerState) -> dict[str, Any]:
    from .dataset import load_frames

    res = run_code(state["code"], load_frames())
    return {
        "stdout": res.stdout,
        "result": res.result,
        "error": res.error,
        "exec_log": [
            {
                "attempt": str(state["attempts"]),
                "code": state["code"],
                "ok": str(res.ok),
                "stdout": res.stdout[:600],
                "error": res.error[-400:],
                "result": res.result[:600],
            }
        ],
    }


def _build_review(model: BaseChatModel) -> Any:
    def review(state: ProgrammerState) -> dict[str, Any]:
        # 実行結果を必ず渡す。コードだけを見ても正しさは判定できない。
        #
        # データ概要も渡す。スキーマを知らないと
        # 「正しい列を使ったか」を判定できず、
        # 例外が出なければ通す判定しかできなくなる。
        user = (
            f"タスク:\n{state['task']}\n\n"
            f"使えるデータ:\n{state['data_summary']}\n\n"
            f"コード:\n```python\n{state['code']}\n```\n\n"
            f"標準出力:\n{state['stdout'] or '(なし)'}\n\n"
            f"result:\n{state['result'] or '(なし)'}\n\n"
            f"エラー:\n{state['error'] or '(なし)'}"
        )
        if state["advice"]:
            joined = "\n".join(f"- {a}" for a in state["advice"])
            user += f"\n\nすでに出した指摘（繰り返さない）:\n{joined}"

        verdict = _invoke_text(model, REVIEW_SYSTEM_PROMPT, user)
        first = verdict.strip().splitlines()[0].upper() if verdict.strip() else ""
        ok = "PASS" in first
        if ok:
            return {"completed": True}
        return {"completed": False, "advice": [verdict]}

    return review


def should_retry(state: ProgrammerState) -> str:
    if state["completed"]:
        return "end"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "end"
    return "retry"


def build_graph(model: BaseChatModel) -> CompiledStateGraph[Any, Any, Any]:
    g: StateGraph[Any, Any, Any, Any] = StateGraph(ProgrammerState)
    g.add_node("generate_code", _build_generate(model))
    g.add_node("execute", _execute)
    g.add_node("review", _build_review(model))

    g.add_edge(START, "generate_code")
    g.add_edge("generate_code", "execute")
    g.add_edge("execute", "review")
    g.add_conditional_edges("review", should_retry, {"retry": "generate_code", "end": END})
    return g.compile()


def initial_state(task: str, data_summary: str) -> ProgrammerState:
    return {
        "task": task,
        "data_summary": data_summary,
        "code": "",
        "attempts": 0,
        "completed": False,
        "exec_log": [],
        "advice": [],
        "stdout": "",
        "result": "",
        "error": "",
    }
