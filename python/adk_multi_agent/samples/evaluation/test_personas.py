"""User Simulation のペルソナとシナリオの前提を固定する。モデルは呼ばない。"""

from __future__ import annotations

import json

import pytest
from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.adk.evaluation.simulation.llm_backed_user_simulator_prompts import (
    get_llm_backed_user_simulator_prompt,
)
from google.adk.evaluation.simulation.pre_built_personas import (
    get_default_persona_registry,
)
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator
from google.genai import types

from samples.evaluation.personas import ADVERSARIAL, SCENARIOS, build
from samples.evaluation.test_evaluation import HERE


def test_prebuilt_personas_have_no_adversarial_user() -> None:
    registry = get_default_persona_registry()
    for persona_id in ("EXPERT", "NOVICE", "EVALUATOR"):
        assert registry.get_persona(persona_id).id == persona_id
    with pytest.raises(NotFoundError, match="ADVERSARIAL"):
        registry.get_persona("ADVERSARIAL")


def test_violation_rubrics_grade_the_simulator_not_the_agent() -> None:
    """利用者役への指示には振る舞いが入り、違反の基準は入らない。基準は利用者役の出来を採点する側で使う。"""
    prompt = get_llm_backed_user_simulator_prompt(
        conversation_plan=SCENARIOS["adversarial_prompt_leak"].conversation_plan,
        conversation_history="",
        stop_signal="</finished>",
        user_persona=ADVERSARIAL,
    )
    behavior = ADVERSARIAL.behaviors[0]
    assert behavior.behavior_instructions[0] in prompt
    assert behavior.violation_rubrics[0] not in prompt


def test_reference_metrics_cannot_score_scenario_cases() -> None:
    """シナリオのケースには期待値が無いので、ツール呼び出しの一致は例外になり、adk eval では未評価になる。"""
    actual = Invocation(
        user_content=types.Content(role="user", parts=[types.Part(text="天気どう？")]),
        intermediate_data=IntermediateData(tool_uses=[]),
    )
    with pytest.raises(ValueError, match="expected_invocations"):
        TrajectoryEvaluator(threshold=1.0).evaluate_invocations(
            actual_invocations=[actual], expected_invocations=None
        )


def test_go_persona_fixture_is_up_to_date() -> None:
    fixture = (
        HERE.parents[3]
        / "go/internal/evalharness/adkeval/testdata/persona.evalset.json"
    )
    assert json.loads(fixture.read_text()) == json.loads(
        build().model_dump_json(exclude_none=True)
    )
