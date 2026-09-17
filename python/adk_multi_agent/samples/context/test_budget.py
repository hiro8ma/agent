"""コンテキスト予算と汚染対策の検査。

見るのは 3 つ。

    予算        比率の合計が 1.0 でなければ組み立て時に落ちる
    間引き      古いツール結果を落とし、最後の利用者発話は必ず残す
    ADK の実物  Compaction と Context Cache と State スコープの実際の形
"""

from __future__ import annotations

import pytest
from google.adk import Agent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.models import LlmRequest
from google.adk.sessions.state import State
from google.genai import types
from pydantic import ValidationError

from samples.context.budget import (
    TRIM_REPORT_KEY,
    ContextBudget,
    content_tokens,
    drop_stale_tool_results,
    enforce_budget,
    trim_to_limit,
)


class FakeCallbackContext:
    """CallbackContext の代わり。コールバックが触るのは state だけ。"""

    def __init__(self) -> None:
        self.state: dict = {}


def text(role: str, body: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part(text=body)])


def tool_result(name: str, payload: dict) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(name=name, response=payload)
            )
        ],
    )


def test_budget_shares_must_sum_to_one():
    """合計が 1.0 でない割り当ては、運用に入る前に落とす。"""
    with pytest.raises(ValueError, match="合計"):
        ContextBudget(shares={"history": 0.5, "output": 0.2})
    with pytest.raises(ValueError):
        ContextBudget(total_tokens=0)


def test_budget_limits_match_the_plan():
    """128K 運用の割り当てが教材の表と一致するか。"""
    b = ContextBudget(total_tokens=128_000)
    assert b.limit("system_instruction") == 12_800
    assert b.limit("history") == 38_400
    assert b.limit("tool_results") == 32_000
    assert b.limit("output") == 19_200
    # 入力側は出力用バッファを引いた残り
    assert b.input_limit == 108_800
    with pytest.raises(KeyError):
        b.limit("unknown")


def test_content_tokens_counts_tool_payloads():
    """ツールの呼び出しと結果も数に入れる。テキストだけ数えると過小評価になる。"""
    assert content_tokens(text("user", "12345678")) == 4
    assert content_tokens(tool_result("search", {"a": 1})) > 0


def test_drop_stale_tool_results_keeps_the_latest():
    """古い結果は落とす。新旧が並ぶと、どちらが現在の値か決められない。"""
    contents = [
        text("user", "在庫ある？"),
        tool_result("stock", {"status": "in_stock"}),
        text("model", "在庫があります"),
        text("user", "本当に？"),
        tool_result("stock", {"status": "out_of_stock"}),
    ]
    kept = drop_stale_tool_results(contents, keep=1)
    payloads = [
        p.function_response.response
        for c in kept
        for p in c.parts or []
        if p.function_response is not None
    ]
    assert payloads == [{"status": "out_of_stock"}]
    # ツール結果以外はそのまま残り、順序も変わらない
    assert [c for c in kept if c in contents] == kept
    assert len(kept) == 4
    with pytest.raises(ValueError):
        drop_stale_tool_results(contents, keep=-1)


def test_trim_to_limit_always_keeps_the_last_turn():
    """上限を超えていても、最後の利用者発話だけは残す。"""
    contents = [
        text("user", "あ" * 100),
        text("model", "い" * 100),
        text("user", "う" * 100),
    ]
    kept = trim_to_limit(contents, limit=1)
    assert kept == [contents[-1]]

    # 収まる範囲では新しい順に残る
    kept = trim_to_limit(contents, limit=120)
    assert kept == contents[1:]
    assert trim_to_limit([], limit=10) == []


def test_enforce_budget_rewrites_the_request_and_reports():
    """コールバックは要求を書き換え、削った量を temp: の鍵へ残す。"""
    budget = ContextBudget(total_tokens=200)
    callback = enforce_budget(budget, keep_tool_results=1)
    request = LlmRequest(
        contents=[
            tool_result("stock", {"status": "in_stock"}),
            text("model", "え" * 300),
            tool_result("stock", {"status": "out_of_stock"}),
            text("user", "結局どっち？"),
        ]
    )
    ctx = FakeCallbackContext()

    assert callback(ctx, request) is None  # 応答は作らず、要求だけ直す

    report = ctx.state[TRIM_REPORT_KEY]
    assert report["contents_after"] < report["contents_before"]
    assert report["tokens_after"] <= report["input_limit"]
    assert TRIM_REPORT_KEY.startswith(State.TEMP_PREFIX)

    payloads = [
        p.function_response.response
        for c in request.contents
        for p in c.parts or []
        if p.function_response is not None
    ]
    assert payloads in ([], [{"status": "out_of_stock"}])
    assert request.contents[-1].parts[0].text == "結局どっち？"


def test_state_scopes_exist():
    """スコープ付きの鍵は接頭辞で決まる。分離の手段として使う。"""
    assert (State.APP_PREFIX, State.USER_PREFIX, State.TEMP_PREFIX) == (
        "app:",
        "user:",
        "temp:",
    )


def test_compaction_requires_both_token_params():
    """token_threshold だけ設定しても効かない。組み立て時に落ちる。"""
    ok = EventsCompactionConfig(compaction_interval=3, overlap_size=1)
    assert ok.token_threshold is None

    with pytest.raises(ValidationError, match="must be set together"):
        EventsCompactionConfig(
            compaction_interval=3, overlap_size=1, token_threshold=50_000
        )

    both = EventsCompactionConfig(
        compaction_interval=3,
        overlap_size=1,
        token_threshold=50_000,
        event_retention_size=10,
    )
    assert both.token_threshold == 50_000


def test_app_accepts_compaction_and_cache_config():
    """App に載せる設定は 3 つ。Compaction は履歴、Cache はコストに効く。"""
    app = App(
        name="context_demo",
        root_agent=Agent(name="demo", model="gemini-3.8-flash", instruction="x"),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=3, overlap_size=1
        ),
        context_cache_config=ContextCacheConfig(min_tokens=4096),
    )
    assert app.events_compaction_config.compaction_interval == 3
    assert app.context_cache_config.min_tokens == 4096
    # 既定値も押さえる。既定のまま使うと 10 回ごとに作り直し、寿命は 30 分
    default_cache = ContextCacheConfig()
    assert (default_cache.cache_intervals, default_cache.ttl_seconds) == (10, 1800)
