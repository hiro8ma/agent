"""Go の adkeval の採点が、ADK の採点と同じ値になるかを突き合わせる入力と期待値を作る。

    uv run python -m samples.evaluation.parity > ../../go/internal/evalharness/adkeval/testdata/parity.json

Go のテストはこのファイルを読み、同じ入力で同じ値になるかを見る。
"""

from __future__ import annotations

import json
import sys

from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.adk.evaluation.eval_metrics import EvalMetric, ToolTrajectoryCriterion
from google.adk.evaluation.trajectory_evaluator import TrajectoryEvaluator
from google.genai import types

from samples.evaluation.ja_response_match import bigram_f1

BIGRAM_CASES = [
    ("東京の天気は晴れで、気温は25度です。", "東京の天気は晴れで、気温は25度です。"),
    ("東京の天気は雨で、気温は25度です。", "東京の天気は晴れで、気温は25度です。"),
    ("大阪は雪、気温は25度でした。", "東京の天気は晴れで、気温は25度です。"),
    ("東京は晴れ。", "東京の天気は晴れで、気温は25度です。"),
    ("東京は２５度。", "東京は25度"),
    ("ｶﾀｶﾅのコーヒー", "カタカナのコーヒー"),
    ("がき", "がき"),
    ("snake_case と CamelCase", "snakecase と camelcase"),
    ("晴れ☀️です", "晴れです"),
    ("", "東京"),
    ("東", "東京"),
]

CALL_CASES = [
    (
        "同じ",
        [("get_weather", {"city": "tokyo"})],
        [("get_weather", {"city": "tokyo"})],
    ),
    (
        "引数の大小文字",
        [("get_weather", {"city": "Tokyo"})],
        [("get_weather", {"city": "tokyo"})],
    ),
    ("整数と小数", [("search", {"n": 2.0})], [("search", {"n": 2})]),
    (
        "余分な呼び出し",
        [("a", {}), ("get_weather", {"city": "tokyo"})],
        [("get_weather", {"city": "tokyo"})],
    ),
    ("順番違い", [("b", {}), ("a", {})], [("a", {}), ("b", {})]),
    ("足りない", [("a", {})], [("a", {}), ("b", {})]),
    ("呼ばないことを期待", [], []),
    ("呼ばないはずが呼んだ", [("a", {})], []),
    ("入れ子の引数", [("f", {"x": {"y": [1, 2]}})], [("f", {"x": {"y": [1, 2]}})]),
]

MATCH_TYPES = ["EXACT", "IN_ORDER", "ANY_ORDER"]


def invocation(calls: list[tuple[str, dict]]) -> Invocation:
    return Invocation(
        user_content=types.Content(role="user", parts=[types.Part(text="q")]),
        intermediate_data=IntermediateData(
            tool_uses=[types.FunctionCall(name=n, args=a) for n, a in calls]
        ),
    )


def trajectory(actual, expected, match_type: str) -> float:
    criterion = ToolTrajectoryCriterion(threshold=1.0, match_type=match_type)
    metric = EvalMetric(
        metric_name="tool_trajectory_avg_score", threshold=1.0, criterion=criterion
    )
    result = TrajectoryEvaluator(eval_metric=metric).evaluate_invocations(
        actual_invocations=[invocation(actual)],
        expected_invocations=[invocation(expected)],
    )
    return result.overall_score


def build() -> dict:
    return {
        "bigram": [
            {"candidate": c, "reference": r, "score": bigram_f1(c, r)}
            for c, r in BIGRAM_CASES
        ],
        "trajectory": [
            {
                "name": name,
                "match_type": mt,
                "actual": [{"name": n, "args": a} for n, a in actual],
                "expected": [{"name": n, "args": a} for n, a in expected],
                "score": trajectory(actual, expected, mt),
            }
            for name, actual, expected in CALL_CASES
            for mt in MATCH_TYPES
        ],
    }


if __name__ == "__main__":
    json.dump(build(), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
