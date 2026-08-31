"""ヘルプデスクエージェントを画面から確認するための HTTP サーバー。

グラフの実行はターミナルのログでは追いにくい。
とくに動的並列では、どのサブタスクが何回やり直したかが混ざって流れる。
実行の各段を構造化して返し、画面で並べて見られるようにする。

逐次版と並列版の両方を同じ形で返す。切り替えて比べられる。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import graph as sequential_graph
from . import parallel_graph
from . import parallel_runner, runner

app = FastAPI(title="helpdesk agent")

# 開発用に画面のオリジンからの取得を許す。本番では許可先を設定から与える。
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class AskRequest(BaseModel):
    query: str
    # parallel なら Send API の動的並列、sequential なら 1 つずつ。
    mode: str = "parallel"
    # 鍵が無い環境でも動かせるよう、既定は決定的な代替モデルにする。
    use_fake: bool = True


@app.get("/api/graph")
def graph_shapes() -> dict[str, str]:
    """両方のグラフ構造を mermaid で返す。API キー不要。"""
    return {
        "sequential": sequential_graph.draw_mermaid(),
        "parallel": parallel_graph.draw_mermaid(),
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    """質問を投げ、各段の記録つきで結果を返す。"""

    start = time.perf_counter()

    if req.mode == "sequential":
        state = runner.run(req.query, use_fake=req.use_fake)
        elapsed = (time.perf_counter() - start) * 1000
        # 逐次版はサブタスクごとの試行回数を持たない。持っている情報だけ返す。
        return {
            "mode": "sequential",
            "elapsedMs": round(elapsed, 1),
            "subtasks": state.get("subtasks", []),
            "results": [
                {"index": i, "subtask": s, "answer": a, "attempts": None, "completed": True}
                for i, (s, a) in enumerate(
                    zip(state.get("subtasks", []), state.get("sub_answers", []))
                )
            ],
            "finalAnswer": state.get("final_answer", ""),
            "history": state.get("history", []),
        }

    state = parallel_runner.run(req.query, use_fake=req.use_fake)
    elapsed = (time.perf_counter() - start) * 1000
    results = sorted(state.get("subtask_results", []), key=lambda r: r["index"])
    return {
        "mode": "parallel",
        "elapsedMs": round(elapsed, 1),
        "subtasks": state.get("subtasks", []),
        "results": results,
        "finalAnswer": state.get("final_answer", ""),
        "history": [],
    }


class AnalyzeRequest(BaseModel):
    task: str
    use_fake: bool = True


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    """データ分析エージェントを走らせ、各試行のコードと実行結果を返す。

    ダッシュボードは「読み取る材料」を出すが、示唆までは出さない。
    ここでは何を実行して何が返ったかを全部出す。
    生成されたコードを見ずに結果だけ信じるのは、
    根拠を確かめずに数字を受け取るのと同じになる。
    """

    import time

    from ..analyst.fake import FakeAnalystModel
    from ..analyst.graph import build_graph, initial_state

    start = time.perf_counter()
    model: Any = FakeAnalystModel()
    if not req.use_fake:
        from core.providers.factory import select_provider

        model = select_provider()

    app_graph = build_graph(model)
    state = app_graph.invoke(initial_state(req.task), {"recursion_limit": 60})
    results = sorted(state["results"], key=lambda r: r["index"])
    return {
        "elapsedMs": round((time.perf_counter() - start) * 1000, 1),
        "plan": state["plan"],
        "results": results,
        "report": state["report"],
    }


@app.get("/analyst", response_class=HTMLResponse)
def analyst_page() -> str:
    return (Path(__file__).parent.parent.parent / "web" / "analyst.html").read_text(
        encoding="utf-8"
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """画面を返す。ビルド不要の 1 枚もの。"""
    return (Path(__file__).parent.parent.parent / "web" / "helpdesk.html").read_text(
        encoding="utf-8"
    )
