"""長い Session を要約する App 設定。"""

from __future__ import annotations

from google.adk import Agent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import BaseLlm, Gemini

MODEL = "gemini-3.8-flash"


def build_compaction_app(
    *,
    model: str | BaseLlm = MODEL,
    summarizer: BaseEventsSummarizer | None = None,
    compaction_interval: int = 20,
    overlap_size: int = 2,
    token_threshold: int = 30_000,
    event_retention_size: int = 10,
) -> App:
    """Invocation 数と直近の入力トークン数で圧縮する App を作る。"""
    agent = Agent(
        name="support_agent",
        model=model,
        instruction="顧客サポートを行い、確認済みの事実だけを回答してください。",
    )
    return App(
        name="compaction_demo",
        root_agent=agent,
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=compaction_interval,
            overlap_size=overlap_size,
            token_threshold=token_threshold,
            event_retention_size=event_retention_size,
            summarizer=summarizer or LlmEventSummarizer(llm=Gemini(model=MODEL)),
        ),
    )
