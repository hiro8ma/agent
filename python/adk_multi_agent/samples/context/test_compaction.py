"""Compaction が Session と次のモデル入力へ与える差を固定する。"""

from collections.abc import AsyncGenerator

from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events import Event
from google.adk.events.event_actions import EventActions, EventCompaction
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from samples.context.compaction import build_compaction_app


class RecordingModel(BaseLlm):
    requests: list[list[str]] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        texts = [
            part.text
            for content in llm_request.contents or []
            for part in content.parts or []
            if part.text
        ]
        self.requests = self.requests + [texts]
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="回答")])
        )


class FixedSummarizer(BaseEventsSummarizer):
    async def maybe_summarize_events(self, *, events: list[Event]) -> Event | None:
        if not events:
            return None
        return Event(
            author="user",
            invocation_id=Event.new_id(),
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=events[0].timestamp,
                    end_timestamp=events[-1].timestamp,
                    compacted_content=types.Content(
                        role="model", parts=[types.Part(text="圧縮済みの履歴")]
                    ),
                )
            ),
        )


def test_compaction_app_keeps_both_trigger_settings():
    app = build_compaction_app()
    config = app.events_compaction_config

    assert config is not None
    assert config.compaction_interval == 20
    assert config.overlap_size == 2
    assert config.token_threshold == 30_000
    assert config.event_retention_size == 10
    assert isinstance(config.summarizer, LlmEventSummarizer)


async def test_runner_keeps_raw_events_but_replaces_the_next_model_history():
    model = RecordingModel(model="recording")
    app = build_compaction_app(
        model=model,
        summarizer=FixedSummarizer(),
        compaction_interval=2,
        overlap_size=0,
        token_threshold=1_000_000,
        event_retention_size=0,
    )
    runner = InMemoryRunner(app=app)
    await runner.session_service.create_session(
        app_name=app.name, user_id="user", session_id="session"
    )

    for message in ("最初の依頼", "二番目の依頼", "三番目の依頼"):
        async for _ in runner.run_async(
            user_id="user",
            session_id="session",
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        ):
            pass

    session = await runner.session_service.get_session(
        app_name=app.name, user_id="user", session_id="session"
    )
    assert session is not None
    assert len([event for event in session.events if event.actions.compaction]) == 1
    assert any(
        part.text == "最初の依頼"
        for event in session.events
        if event.content and event.content.parts
        for part in event.content.parts
    )

    third_request = model.requests[2]
    assert any("圧縮済みの履歴" in text for text in third_request)
    assert "三番目の依頼" in third_request
    assert "最初の依頼" not in third_request
    assert "二番目の依頼" not in third_request
