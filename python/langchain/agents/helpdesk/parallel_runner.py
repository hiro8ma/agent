"""動的並列版の実行ヘルパー。

逐次版（runner.py）と同じ入口を持たせて、画面から切り替えられるようにする。
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel

from .parallel_graph import build_graph, initial_state


def _select_model(use_fake: bool) -> tuple[BaseChatModel, bool]:
    if not use_fake and os.environ.get("OPENAI_API_KEY"):
        from core.providers.factory import select_provider

        return select_provider(), False
    from .fake import FakeChatModel

    return FakeChatModel(), True


def stream(query: str, use_fake: bool = False) -> Iterator[tuple[str, dict[str, Any]]]:
    """各ステップの (ノード名, 状態の更新) を流す。"""
    model, _ = _select_model(use_fake)
    app = build_graph(model)
    for chunk in app.stream(initial_state(query), {"recursion_limit": 100}):
        yield from chunk.items()


def run(query: str, use_fake: bool = False) -> dict[str, Any]:
    model, _ = _select_model(use_fake)
    app = build_graph(model)
    return dict(app.invoke(initial_state(query), {"recursion_limit": 100}))


def timed_run(query: str, use_fake: bool = False) -> dict[str, Any]:
    """実行時間つきで走らせる。逐次版との比較に使う。"""
    start = time.perf_counter()
    state = run(query, use_fake)
    state["_elapsed_ms"] = (time.perf_counter() - start) * 1000
    return state
