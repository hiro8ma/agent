"""Compaction の発火点と要約の範囲を実測で固定する。

台本モデルと、要約に渡ったイベントを記録する要約器で測る。実モデルは使わない。
保留中の呼び出しがある場合の挙動は ADK 2.2.0 時点の観測で、不具合の可能性がある。
ADK を上げて変わったら、この検査が落ちて気づける。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events import Event
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import LongRunningFunctionTool
from google.genai import types
from pydantic import ValidationError

from samples.compaction.config import PRESETS, compaction_config


class Scripted(BaseLlm):
    """台本どおりに答えるモデル。応答ごとに prompt_token_count を申告できる。"""

    replies: list = []
    prompt_tokens: int | None = None
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        i = self.calls
        self.calls = i + 1
        reply = self.replies[i] if i < len(self.replies) else f"応答{i}"
        content = (
            reply
            if isinstance(reply, types.Content)
            else types.Content(role="model", parts=[types.Part(text=reply)])
        )
        response = LlmResponse(content=content)
        if self.prompt_tokens is not None:
            response.usage_metadata = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=self.prompt_tokens
            )
        yield response


class Recording(LlmEventSummarizer):
    """要約に渡ったイベントを記録する。"""

    def __init__(self) -> None:
        super().__init__(
            llm=Scripted(model="summarizer", replies=[f"要約{i}" for i in range(20)])
        )
        self.batches: list[list[Event]] = []

    async def maybe_summarize_events(self, *, events: list[Event]) -> Event | None:
        self.batches.append(events)
        return await super().maybe_summarize_events(events=events)


def user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def text_of(event: Event) -> str:
    part = event.content.parts[0] if event.content and event.content.parts else None
    if part is None:
        return ""
    if part.function_call:
        return "CALL"
    return part.text or ""


def shape(events: list[Event]) -> str:
    """セッションの並びを U / M / CALL / [圧縮] で表す。"""
    out = []
    for e in events:
        if e.actions and e.actions.compaction:
            out.append("[圧縮]")
        elif e.author == "user":
            out.append("U")
        elif text_of(e) == "CALL":
            out.append("CALL")
        else:
            out.append("M")
    return " ".join(out)


async def run(app: App, turns: list[str]):
    runner = InMemoryRunner(app=app)
    await runner.session_service.create_session(
        app_name=app.name, user_id="u", session_id="s"
    )
    for text in turns:
        async for _ in runner.run_async(
            user_id="u", session_id="s", new_message=user(text)
        ):
            pass
    return await runner.session_service.get_session(
        app_name=app.name, user_id="u", session_id="s"
    )


# --- 設定 -----------------------------------------------------------------


def test_token_threshold_requires_retention_size() -> None:
    with pytest.raises(ValidationError, match="must be set together"):
        EventsCompactionConfig(
            compaction_interval=10, overlap_size=1, token_threshold=100
        )


def test_presets_follow_the_use_case_table() -> None:
    cfg = compaction_config("support")
    assert (cfg.compaction_interval, cfg.overlap_size) == PRESETS["support"]
    assert cfg.summarizer is None


def test_unknown_use_case_is_rejected() -> None:
    with pytest.raises(ValueError, match="未知の用途"):
        compaction_config("unknown")


# --- Invocation 間隔 ------------------------------------------------------


async def test_interval_fires_after_the_invocation_ends() -> None:
    recorder = Recording()
    app = App(
        name="interval",
        root_agent=Agent(name="a", model=Scripted(model="agent"), instruction="x"),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=2, overlap_size=1, summarizer=recorder
        ),
    )
    session = await run(app, ["t0", "t1", "t2", "t3"])

    # 2 回目と 4 回目の Invocation が終わった後に、末尾へ圧縮が入る。
    assert shape(session.events) == "U M U M [圧縮] U M U M [圧縮]"

    first = {e.invocation_id for e in recorder.batches[0]}
    second = {e.invocation_id for e in recorder.batches[1]}
    assert len(first) == 2
    # overlap_size=1 の分、前回の範囲の最後の Invocation をもう一度要約する。
    assert len(second) == 3
    assert len(first & second) == 1


async def test_default_summarizer_is_the_agents_own_model() -> None:
    model = Scripted(model="agent", replies=["応答", "要約文"])
    config = EventsCompactionConfig(compaction_interval=1, overlap_size=0)
    await run(
        App(
            name="default",
            root_agent=Agent(name="a", model=model, instruction="x"),
            events_compaction_config=config,
        ),
        ["t0"],
    )
    assert isinstance(config.summarizer, LlmEventSummarizer)
    assert config.summarizer._llm is model


# --- トークン閾値 ---------------------------------------------------------


async def token_session():
    recorder = Recording()
    app = App(
        name="token",
        root_agent=Agent(
            name="a", model=Scripted(model="agent", prompt_tokens=5000), instruction="x"
        ),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=1000,
            overlap_size=0,
            token_threshold=100,
            event_retention_size=1,
            summarizer=recorder,
        ),
    )
    session = await run(app, ["質問0", "質問1", "質問2"])
    return session, recorder


async def test_token_threshold_fires_before_the_model_call() -> None:
    """発火点は Invocation の終了後だけではない。モデル呼び出しの直前でも走る。"""
    session, _ = await token_session()
    assert "U [圧縮] M" in shape(session.events)


async def test_token_threshold_keeps_resummarizing_the_previous_summary() -> None:
    """申告値が閾値を越えている限り、判定のたびに前回の要約ごと要約し直す。"""
    _session, recorder = await token_session()
    assert len(recorder.batches) == 5
    for batch in recorder.batches[1:]:
        assert len(batch) == 2
        assert batch[0].author == "model"
        assert text_of(batch[0]).startswith("要約")


# --- 保留中の呼び出し -----------------------------------------------------


async def test_pending_call_freezes_the_sliding_window() -> None:
    """保留中の呼び出しの手前で止まり、以降は同じ 1 イベントを要約し直し続ける。

    ADK 2.2.0 時点の観測。後ろの Invocation は圧縮されず、要約の呼び出しだけが増える。
    """

    def wait_approval(item: str) -> None:
        """人の承認を待つ。結果は返さず、呼び出しを保留のまま残す。"""

    call = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(
                    name="wait_approval", args={"item": "x"}
                )
            )
        ],
    )
    recorder = Recording()
    app = App(
        name="pending",
        root_agent=Agent(
            name="a",
            model=Scripted(model="agent", replies=["応答0", call, "応答2", "応答3"]),
            instruction="x",
            tools=[LongRunningFunctionTool(wait_approval)],
        ),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=2, overlap_size=0, summarizer=recorder
        ),
    )
    session = await run(app, [f"質問{i}" for i in range(6)])

    assert [text_of(e) for e in recorder.batches[0]] == ["質問0", "応答0", "質問1"]
    assert all("CALL" not in [text_of(e) for e in b] for b in recorder.batches)
    for batch in recorder.batches[1:]:
        assert [text_of(e) for e in batch] == ["質問1"]

    raw = [e for e in session.events if not (e.actions and e.actions.compaction)]
    assert len(raw) == 12
