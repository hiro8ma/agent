"""Memory Bank へ書く経路と、CLI の URI の解釈を固定する。

Agent Engine を作らずに確かめられる、純粋な判定関数だけを対象にする。
内部関数なので ADK の版が上がると消えうる。消えたらこの検査が落ちて気づける。
"""

from __future__ import annotations

import pytest
from google.adk.cli.service_registry import _parse_agent_engine_kwargs
from google.adk.memory.vertex_ai_memory_bank_service import (
    _should_use_generate_memories,
)


@pytest.mark.parametrize(
    ("custom_metadata", "uses_generate"),
    [
        (None, False),
        ({}, False),
        ({"generation_trigger_config": {}}, False),
        ({"wait_for_completion": True}, True),
        ({"disable_consolidation": True}, True),
        ({"metadata": {"k": "v"}}, True),
    ],
)
def test_session_ingest_path_switches_on_generate_only_keys(
    custom_metadata: dict | None, uses_generate: bool
) -> None:
    """add_session_to_memory の既定は ingest_events で、GenerateMemories ではない。

    保存直後に検索へ反映させたくて wait_for_completion を渡すと、
    待つだけのつもりが経路ごと generate_memories に切り替わる。
    """
    assert _should_use_generate_memories(custom_metadata) is uses_generate


def test_agentengine_uri_accepts_the_full_resource_name() -> None:
    params = _parse_agent_engine_kwargs(
        "projects/p/locations/us-central1/reasoningEngines/123", None
    )
    assert params == {
        "project": "p",
        "location": "us-central1",
        "agent_engine_id": "123",
    }


def test_agentengine_uri_rejects_a_malformed_resource_name() -> None:
    with pytest.raises(ValueError, match="mal-formatted"):
        _parse_agent_engine_kwargs("projects/p/reasoningEngines/123", None)


def test_agentengine_uri_rejects_an_empty_value() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        _parse_agent_engine_kwargs("", None)
