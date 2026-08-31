"""Plan-and-Execute を動的並列で組んだ版。

`graph.py` の逐次版と同じ題材を、Send API で並列に走らせる。
両方を残して比べられるようにしてある。

## 逐次版との違い

```
逐次版   plan → execute → reflect → advance → execute → ... → synthesize
         current_subtask_index を進めて 1 つずつ処理する
         サブタスクが N 個なら実行時間も N 倍になる

並列版   create_plan → Send で N 個同時に execute_subtask → create_answer
         サブタスクごとに独立して走り、自己修正もその中で完結する
```

## なぜ Send API が要るか

グラフを組み立てる時点では、サブタスクが何個になるか分からない。
計画を作るのは実行時なので、静的なエッジでは書けない。

`add_conditional_edges` が `Send` のリストを返すと、
LangGraph はその数だけノードを起動する。**エッジの数を実行時に決める**唯一の方法になる。

## 自己修正の上限

3 回で打ち切る。

サブタスクが失敗する主因は、**そもそもドキュメントに情報が無い**ことになる。
これは何回やり直しても直らない。
「リトライで直る失敗」と「データが無い失敗」は別物で、
後者に上限を置かないと無限ループになる。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from .prompt import (
    ANSWER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REFLECT_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SYNTHESIZE_SYSTEM_PROMPT,
)
from .tools import search_hybrid, search_manual, search_qa

# やり直しの上限。教材と同じ 3 回。
MAX_ATTEMPTS = 3


class SubtaskResult(TypedDict):
    """1 つのサブタスクの結果。

    index を持たせるのは、並列で戻る順序が計画の順序と一致しないため。
    逐次版では順番に積めばよかったが、並列では復元に添字が要る。
    """

    index: int
    subtask: str
    answer: str
    attempts: int
    completed: bool
    tool_log: list[dict[str, str]]
    advice: list[str]


class MainState(TypedDict):
    """メイングラフの状態。

    subtask_results だけ `operator.add` で積む。
    並列に走った各サブグラフが、それぞれ 1 件を append する。
    """

    query: str
    subtasks: list[str]
    subtask_results: Annotated[list[SubtaskResult], operator.add]
    final_answer: str


class SubtaskState(TypedDict):
    """サブグラフの状態。

    Send で渡される内容がそのまま入る。
    メイングラフの状態とは別で、サブタスク同士は互いを見ない。
    見えると、あるサブタスクの検索結果が別のサブタスクの回答に混ざる。
    """

    query: str
    subtask: str
    index: int
    attempts: int
    draft: str
    critique: str
    completed: bool
    # tool_log は各試行で「どのツールを何で引いて何が返ったか」。
    #
    # 自己修正はツールの実行結果と回答の両方を見る必要がある。
    # 回答だけを見ると「情報が見つからなかった」を判定できず、
    # それらしい文章が返ってきた時点で OK になる。
    tool_log: Annotated[list[dict[str, str]], operator.add]
    # advice は過去のアドバイス全部。
    #
    # 直前の 1 件だけを渡すと、同じ助言が繰り返される。
    # 3 回やり直しても同じツールを試し続ける形になる。
    advice: Annotated[list[str], operator.add]
    # selected_tool と retrieved はノード間の受け渡し。
    #
    # TypedDict に宣言しないキーは LangGraph が黙って捨てる。
    # 宣言を忘れると、書いた側はエラーにならず、読む側が既定値に倒れる。
    # 実際、選んだツールが伝わらず常に既定のツールが走り、
    # 検索結果も自己修正へ届いていなかった。どちらも例外は出ない。
    selected_tool: str
    retrieved: str
    subtask_results: Annotated[list[SubtaskResult], operator.add]


def _invoke_text(model: BaseChatModel, system: str, user: str) -> str:
    response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = response.content
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "\n".join(parts).strip()


# --- メイングラフのノード ---


def _build_create_plan(model: BaseChatModel) -> Any:
    def create_plan(state: MainState) -> dict[str, Any]:
        text = _invoke_text(model, PLANNER_SYSTEM_PROMPT, f"Inquiry:\n{state['query']}")
        subtasks = [line.strip("-• \t") for line in text.splitlines() if line.strip()]
        if not subtasks:
            subtasks = [state["query"]]
        return {"subtasks": subtasks}

    return create_plan


def fan_out(state: MainState) -> list[Send]:
    """計画の要素数だけ Send を返す。

    グラフの組み立て時にはこの数が分からない。
    計画を作った直後に初めて決まるため、静的なエッジでは表現できない。
    """
    return [
        Send(
            "execute_subtask",
            {
                "query": state["query"],
                "subtask": subtask,
                "index": index,
                "attempts": 0,
                "draft": "",
                "critique": "",
                "completed": False,
                "tool_log": [],
                "advice": [],
                "selected_tool": "",
                "retrieved": "",
                "subtask_results": [],
            },
        )
        for index, subtask in enumerate(state["subtasks"])
    ]


def _build_create_answer(model: BaseChatModel) -> Any:
    def create_answer(state: MainState) -> dict[str, Any]:
        # 並列で戻るため順序が保証されない。index で並べ直す。
        results = sorted(state["subtask_results"], key=lambda r: r["index"])

        # 渡すのはサブタスクの内容と回答だけにする。
        # 対話履歴や検索結果を丸ごと渡すとトークンを食い、
        # かつ最終回答が中間の推測を拾ってしまう。
        joined = "\n\n".join(
            f"[{r['index']}] {r['subtask']}\n{r['answer']}" for r in results
        )

        unresolved = [r for r in results if not r["completed"]]
        user = f"Original inquiry:\n{state['query']}\n\nSubtask answers:\n{joined}"
        if unresolved:
            # 未解決を明示する。伏せると、最終回答が
            # 「分からなかった」を隠して確からしい文章を作ってしまう。
            names = "\n".join(f"- {r['subtask']}" for r in unresolved)
            user += (
                f"\n\nUnresolved subtasks (say so plainly and offer to keep "
                f"investigating; do not guess and do not redirect the user to "
                f"another team):\n{names}"
            )

        final = _invoke_text(model, SYNTHESIZE_SYSTEM_PROMPT, user)
        return {"final_answer": final}

    return create_answer


# --- サブグラフのノード ---


def _build_select_tools(model: BaseChatModel) -> Any:
    def select_tools(state: SubtaskState) -> dict[str, Any]:
        choice = _invoke_text(
            model, ROUTER_SYSTEM_PROMPT, f"Subtask:\n{state['subtask']}"
        ).lower()
        # 想定外の応答はキーワード検索に倒す。
        name = "search_qa" if "search_qa" in choice else "search_manual"
        # やり直しではツールを変える。同じものを引き直しても結果は変わらない。
        #
        # 2 回目は反対側、3 回目はハイブリッドに倒す。
        # 選択で片方に倒すと、選び損ねた側にしかない文書へ到達できない。
        # 最後は両方を統合した検索で拾いにいく。
        if state["attempts"] == 1:
            name = "search_qa" if state.get("selected_tool") == "search_manual" else "search_manual"
        elif state["attempts"] >= 2:
            name = "search_hybrid"
        return {"draft": "", "critique": "", "attempts": state["attempts"] + 1,
                "selected_tool": name}

    return select_tools


def _execute_tools(state: SubtaskState) -> dict[str, Any]:
    name = state.get("selected_tool", "search_manual")
    query = state["subtask"]
    if name == "search_hybrid":
        context = search_hybrid(query)
    elif name == "search_qa":
        context = search_qa(query)
    else:
        context = search_manual(query)
    return {
        "retrieved": context,
        "tool_log": [{"attempt": str(state["attempts"]), "tool": name,
                      "query": query, "result": context}],
    }


def _build_create_subtask_answer(model: BaseChatModel) -> Any:
    def create_subtask_answer(state: SubtaskState) -> dict[str, Any]:
        context = state.get("retrieved", "")
        critique = state.get("critique", "")
        # やり直しのときは前回の指摘を渡す。渡さないと同じ答えを繰り返す。
        user = f"Subtask:\n{state['subtask']}\n\nRetrieved context:\n{context}"
        if critique:
            user += f"\n\nPrevious critique to address:\n{critique}"
        answer = _invoke_text(model, ANSWER_SYSTEM_PROMPT, user)
        return {"draft": answer}

    return create_subtask_answer


def _build_reflect_subtask(model: BaseChatModel) -> Any:
    def reflect_subtask(state: SubtaskState) -> dict[str, Any]:
        # ツールの実行結果を必ず渡す。
        # 回答だけを見ると「情報が見つからなかった」を判定できない。
        context = state.get("retrieved", "")
        past = state.get("advice", [])
        user = (
            f"Subtask:\n{state['subtask']}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Draft answer:\n{state['draft']}"
        )
        if past:
            # 過去のアドバイスを全部渡す。重複した助言を防ぐ。
            joined = "\n".join(f"- {a}" for a in past)
            user += f"\n\nAdvice already given (do not repeat):\n{joined}"
        verdict = _invoke_text(model, REFLECT_SYSTEM_PROMPT, user)
        # プロンプトは 1 行目に `VERDICT: PASS` か `VERDICT: RETRY` を返す約束になる。
        # 別の語で判定すると常に False になり、上限まで回して必ず失敗する。
        # 「毎回リトライして諦める」は、正常動作と区別が付きにくい。
        first = verdict.strip().splitlines()[0].upper() if verdict.strip() else ""
        ok = "PASS" in first
        if ok:
            return {"completed": True, "critique": ""}
        return {"completed": False, "critique": verdict, "advice": [verdict]}

    return reflect_subtask


# 上限に達して未完だったときに残す文言。
#
# draft をそのまま返してはいけない。
# 上限で諦めたときの draft は「それらしいが裏付けの無い文章」であり、
# 最終回答がそれを確からしい材料として扱ってしまう。
# 「見つからなかった」に置き換えて、不確実性を後段へ伝える。
NOT_FOUND_TEMPLATE = "「{subtask}」の回答は見つかりませんでした。"


def _commit(state: SubtaskState) -> dict[str, Any]:
    """サブタスクの結果をメイングラフへ 1 件返す。

    上限に達して未完のまま来ることもある。
    その場合も握り潰さず、答えが出なかったことを明示して返す。
    """
    completed = state["completed"]
    answer = (
        state["draft"]
        if completed
        else NOT_FOUND_TEMPLATE.format(subtask=state["subtask"])
    )
    return {
        "subtask_results": [
            SubtaskResult(
                index=state["index"],
                subtask=state["subtask"],
                answer=answer,
                attempts=state["attempts"],
                completed=state["completed"],
                tool_log=list(state.get("tool_log", [])),
                advice=list(state.get("advice", [])),
            )
        ]
    }


def should_retry(state: SubtaskState) -> str:
    """やり直すか、諦めて返すかを決める。

    上限に達したら completed が False のままでも返す。
    失敗の主因はドキュメントに情報が無いことなので、
    回数を重ねても状況は変わらない。
    """
    if state["completed"]:
        return "commit"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "commit"
    return "retry"


def build_subgraph(model: BaseChatModel) -> CompiledStateGraph[Any, Any, Any]:
    """サブグラフ。1 つのサブタスクを自己修正付きで解く。"""

    sub: StateGraph[Any, Any, Any, Any] = StateGraph(SubtaskState)
    sub.add_node("select_tools", _build_select_tools(model))
    sub.add_node("execute_tools", _execute_tools)
    sub.add_node("create_subtask_answer", _build_create_subtask_answer(model))
    sub.add_node("reflect_subtask", _build_reflect_subtask(model))
    sub.add_node("commit", _commit)

    sub.add_edge(START, "select_tools")
    sub.add_edge("select_tools", "execute_tools")
    sub.add_edge("execute_tools", "create_subtask_answer")
    sub.add_edge("create_subtask_answer", "reflect_subtask")
    sub.add_conditional_edges(
        "reflect_subtask",
        should_retry,
        {"retry": "select_tools", "commit": "commit"},
    )
    sub.add_edge("commit", END)
    return sub.compile()


def _build_execute_subtask(model: BaseChatModel) -> Any:
    """サブグラフを呼び、メイングラフへ返す値を subtask_results だけに絞る。

    サブグラフをそのままノードに埋めると、終了時の状態が丸ごと
    メイングラフへ書き戻される。並列で 3 本走れば query も 3 回書かれ、
    蓄積用に注釈していないフィールドで衝突する。

        InvalidUpdateError: At key 'query': Can receive only one value per step.

    逐次版では書き込みが 1 つずつなので、この衝突は起きない。
    **並列にして初めて出る。**
    返す鍵を絞れば、蓄積用の subtask_results だけがメイングラフに載る。
    """

    subgraph = build_subgraph(model)

    def execute_subtask(state: SubtaskState) -> dict[str, Any]:
        result = subgraph.invoke(state, {"recursion_limit": 50})
        return {"subtask_results": result["subtask_results"]}

    return execute_subtask


def build_graph(model: BaseChatModel) -> CompiledStateGraph[Any, Any, Any]:
    """メイングラフ。計画を作り、サブグラフを動的に並列起動する。"""

    workflow: StateGraph[Any, Any, Any, Any] = StateGraph(MainState)
    workflow.add_node("create_plan", _build_create_plan(model))
    workflow.add_node("execute_subtask", _build_execute_subtask(model))
    workflow.add_node("create_answer", _build_create_answer(model))

    workflow.add_edge(START, "create_plan")
    # Send のリストを返す条件付きエッジ。エッジの数が実行時に決まる。
    workflow.add_conditional_edges("create_plan", fan_out, ["execute_subtask"])
    workflow.add_edge("execute_subtask", "create_answer")
    workflow.add_edge("create_answer", END)
    return workflow.compile()


def initial_state(query: str) -> MainState:
    return {"query": query, "subtasks": [], "subtask_results": [], "final_answer": ""}


def draw_mermaid(model: BaseChatModel | None = None) -> str:
    """グラフ構造を mermaid で返す。API キー不要。"""
    if model is None:
        from .fake import FakeChatModel

        model = FakeChatModel()
    return build_graph(model).get_graph().draw_mermaid()
