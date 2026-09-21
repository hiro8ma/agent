"""日本語の応答を文字の 2-gram で比べる独自の指標。

ADK の response_match_score は ROUGE-1 で、既定の単語分割は英小文字と数字しか残さない。
日本語の応答は数字だけの列になり、「晴れ」と「雨」の違いも、答えの有無も区別できない。

この指標は文字の 2-gram の重なり（F1）を返す。言い回しの近さは測れるが、正しさは測れない。
意味の一致は LLM による判定（final_response_match_v2 やルーブリック）で見る。

評価の設定に次のように書く。

    {
      "criteria": {"ja_response_match": 0.5},
      "custom_metrics": {
        "ja_response_match": {"code_config": {"name": "samples.evaluation.ja_response_match.evaluate"}}
      }
    }
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from google.adk.evaluation.eval_case import ConversationScenario, Invocation
from google.adk.evaluation.eval_metrics import EvalMetric, EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult, PerInvocationResult
from google.genai import types

_SEPARATORS = re.compile(r"[\s\W_]+")


def _text(content: types.Content | None) -> str:
    if content is None:
        return ""
    return "".join(p.text or "" for p in content.parts or [])


def _bigrams(text: str) -> Counter[str]:
    t = _SEPARATORS.sub("", unicodedata.normalize("NFKC", text).lower())
    return Counter(t[i : i + 2] for i in range(len(t) - 1))


def bigram_f1(candidate: str, reference: str) -> float:
    got, want = _bigrams(candidate), _bigrams(reference)
    overlap = sum((got & want).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(got.values())
    recall = overlap / sum(want.values())
    return 2 * precision * recall / (precision + recall)


def evaluate(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: list[Invocation] | None,
    conversation_scenario: ConversationScenario | None = None,
) -> EvaluationResult:
    """ADK は呼ぶ前に eval_metric.threshold を None にするので、合格ラインは criterion から読む。"""
    del conversation_scenario
    threshold = eval_metric.criterion.threshold
    if not expected_invocations:
        return EvaluationResult()

    results = []
    for actual, expected in zip(actual_invocations, expected_invocations, strict=True):
        score = bigram_f1(_text(actual.final_response), _text(expected.final_response))
        status = EvalStatus.PASSED if score >= threshold else EvalStatus.FAILED
        results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
                score=score,
                eval_status=status,
            )
        )
    overall = sum(r.score for r in results) / len(results)
    return EvaluationResult(
        overall_score=overall,
        overall_eval_status=EvalStatus.PASSED
        if overall >= threshold
        else EvalStatus.FAILED,
        per_invocation_results=results,
    )
