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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """画面を返す。ビルド不要の 1 枚もの。"""
    return (Path(__file__).parent.parent.parent / "web" / "helpdesk.html").read_text(
        encoding="utf-8"
    )
