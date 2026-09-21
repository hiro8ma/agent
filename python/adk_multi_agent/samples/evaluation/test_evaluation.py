"""adk eval の評価セットと採点の前提を固定する。モデルは呼ばない。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from google.adk.agents import Agent
from google.adk.cli.cli_eval import get_root_agent
from google.adk.evaluation.custom_metric_evaluator import _CustomMetricEvaluator
from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.adk.evaluation.eval_config import (
    EvalConfig,
    get_eval_metrics_from_config,
    get_evaluation_criteria_or_default,
)
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.final_response_match_v1 import _calculate_rouge_1_scores
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator
from google.genai import types

from samples.evaluation.ja_response_match import bigram_f1

HERE = Path(__file__).parent
EXPECTED = "東京の天気は晴れで、気温は25度です。"


def invocation(text: str, tool_args: dict | None = None) -> Invocation:
    return Invocation(
        user_content=types.Content(
            role="user", parts=[types.Part(text="東京の天気は？")]
        ),
        final_response=types.Content(role="model", parts=[types.Part(text=text)]),
        intermediate_data=IntermediateData(
            tool_uses=[types.FunctionCall(name="get_weather", args=tool_args)]
            if tool_args
            else []
        ),
    )


def test_material_evalset_loads_without_invocation_id() -> None:
    raw = json.loads((HERE / "material.evalset.json").read_text())
    evalset = EvalSet.model_validate(raw)
    [turn] = evalset.eval_cases[0].conversation
    assert turn.intermediate_data.tool_uses[0].args == {"location": "Tokyo"}
    assert turn.invocation_id == ""


@pytest.mark.parametrize(
    ("candidate", "want"),
    [
        (EXPECTED, 1.0),
        ("東京の天気は雨で、気温は25度です。", 1.0),
        ("大阪は雪、気温は25度でした。", 1.0),
        ("東京は晴れ。", 0.0),
    ],
    ids=[
        "同じ文",
        "晴れを雨にした誤答も満点",
        "都市も天気も違う誤答も満点",
        "正しく簡潔な答えは 0 点",
    ],
)
def test_response_match_score_sees_only_digits_in_japanese(
    candidate: str, want: float
) -> None:
    """ROUGE-1 の既定の単語分割は英小文字と数字しか残さない。日本語の文は数字の列になる。"""
    assert _calculate_rouge_1_scores(candidate, EXPECTED).fmeasure == want


def test_default_criteria() -> None:
    """既定の合格ラインは 0.8。日本語では数字が合うかどうかでほぼ決まる。"""
    assert get_evaluation_criteria_or_default(None).criteria == {
        "tool_trajectory_avg_score": 1.0,
        "response_match_score": 0.8,
    }


@pytest.mark.parametrize(
    ("args", "want"),
    [
        ({"location": "Tokyo"}, 1.0),
        ({"location": "tokyo"}, 0.0),
        ({"location": "東京"}, 0.0),
    ],
    ids=[
        "綴りまで同じなら一致",
        "大文字小文字が違うと不一致",
        "日本語の地名だと不一致",
    ],
)
def test_tool_trajectory_compares_args_exactly(args: dict, want: float) -> None:
    """既定の match_type は EXACT で、引数の値も文字列として完全一致を求める。"""
    evaluator = TrajectoryEvaluator(threshold=1.0)
    result = evaluator.evaluate_invocations(
        actual_invocations=[invocation(EXPECTED, args)],
        expected_invocations=[invocation(EXPECTED, {"location": "Tokyo"})],
    )
    assert result.overall_score == want


def test_adk_eval_loads_root_agent_not_app() -> None:
    """adk eval は agent.py の root_agent だけを読む。App の Plugin と Compaction は評価で効かない。"""
    loaded = get_root_agent(str(HERE.parent / "support"))
    assert isinstance(loaded, Agent)


@pytest.mark.parametrize(
    ("candidate", "want_status"),
    [
        (EXPECTED, EvalStatus.PASSED),
        ("大阪は雪、気温は25度でした。", EvalStatus.FAILED),
        ("東京は晴れ、25度です。", EvalStatus.PASSED),
        # 言い回しの近さしか見ないので、誤答が通り、簡潔な正答が落ちる。意味は LLM の判定で見る。
        ("東京の天気は雨で、気温は25度です。", EvalStatus.PASSED),
        ("東京は晴れ。", EvalStatus.FAILED),
    ],
    ids=[
        "同じ文",
        "都市も天気も違う",
        "言い換えた正答",
        "晴れを雨にした誤答は通る",
        "簡潔な正答は落ちる",
    ],
)
async def test_japanese_bigram_metric(candidate: str, want_status: EvalStatus) -> None:
    config = EvalConfig(
        criteria={"ja_response_match": 0.5},
        custom_metrics={
            "ja_response_match": {
                "code_config": {"name": "samples.evaluation.ja_response_match.evaluate"}
            }
        },
    )
    [metric] = get_eval_metrics_from_config(config)
    evaluator = _CustomMetricEvaluator(metric, metric.custom_function_path)
    result = await evaluator.evaluate_invocations(
        [invocation(candidate)], [invocation(EXPECTED)]
    )
    assert result.overall_eval_status == want_status


async def test_custom_metric_receives_no_threshold() -> None:
    """ADK は独自の指標を呼ぶ前に eval_metric.threshold を消す。合格ラインは criterion に残る。"""
    seen = {}

    def spy(eval_metric, actual, expected, scenario):
        seen["threshold"] = eval_metric.threshold
        seen["criterion"] = eval_metric.criterion.threshold
        from google.adk.evaluation.evaluator import EvaluationResult

        return EvaluationResult()

    config = EvalConfig(criteria={"spy": 0.7})
    [metric] = get_eval_metrics_from_config(config)
    evaluator = _CustomMetricEvaluator.__new__(_CustomMetricEvaluator)
    evaluator._eval_metric = metric
    evaluator._metric_function = spy
    await evaluator.evaluate_invocations([invocation(EXPECTED)], [invocation(EXPECTED)])
    assert seen == {"threshold": None, "criterion": 0.7}


def test_bigram_f1_ignores_width_and_punctuation() -> None:
    assert bigram_f1("東京は２５度。", "東京は25度") == 1.0


async def test_multi_turn_feeds_the_agents_own_answers_not_the_expected_ones() -> None:
    """2 ターン目のモデルが受け取る履歴には、評価セットの期待する応答ではなく、1 ターン目の実際の応答が入る。"""
    from collections.abc import AsyncGenerator

    from google.adk.evaluation.evaluation_generator import EvaluationGenerator
    from google.adk.evaluation.simulation.static_user_simulator import (
        StaticUserSimulator,
    )
    from google.adk.models import BaseLlm, LlmRequest, LlmResponse

    class Recorder(BaseLlm):
        histories: list[list[str]] = []

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            del stream
            texts = [
                p.text for c in llm_request.contents for p in c.parts or [] if p.text
            ]
            self.histories = [*self.histories, texts]
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"実際の応答 {len(self.histories)}")],
                )
            )

    turns = [invocation("期待する応答 1"), invocation("期待する応答 2")]
    model = Recorder(model="gemini-3.8-flash")
    await EvaluationGenerator._generate_inferences_from_root_agent(
        root_agent=Agent(name="weather", model=model, instruction="x"),
        user_simulator=StaticUserSimulator(static_conversation=turns),
    )

    second = model.histories[1]
    assert "実際の応答 1" in second
    assert "期待する応答 1" not in second


def test_go_parity_fixture_is_up_to_date() -> None:
    """Go の adkeval が読む突き合わせ用の値が、いまの ADK の採点と一致する。

    ずれたら python -m samples.evaluation.parity で書き直す。
    """
    from samples.evaluation.parity import build

    fixture = HERE.parents[3] / "go/internal/evalharness/adkeval/testdata/parity.json"
    assert json.loads(fixture.read_text()) == json.loads(
        json.dumps(build(), ensure_ascii=False)
    )
