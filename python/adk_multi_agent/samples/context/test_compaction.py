"""Compaction 設定を ADK v2.2.0 の公開 API で固定する。"""

from google.adk.apps.llm_event_summarizer import LlmEventSummarizer

from samples.context.compaction import build_compaction_app


def test_compaction_app_keeps_both_trigger_settings():
    app = build_compaction_app()
    config = app.events_compaction_config

    assert config is not None
    assert config.compaction_interval == 20
    assert config.overlap_size == 2
    assert config.token_threshold == 30_000
    assert config.event_retention_size == 10
    assert isinstance(config.summarizer, LlmEventSummarizer)
