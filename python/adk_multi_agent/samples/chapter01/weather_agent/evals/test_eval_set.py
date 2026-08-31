"""評価セット自体を検査する。

評価セットは評価される側と同じだけ間違える。
未登録の都市を正解に書けば、正しい実装が落ちる。

adk eval は API キーが要るため、ここでは形式と中身だけを見る。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from google.adk.evaluation.eval_set import EvalSet

from samples.chapter01.weather_agent.agent import _WEATHER, get_weather

SET_PATH = Path(__file__).parent / "weather_agent_v1.evalset.json"


@pytest.fixture(scope="module")
def eval_set() -> EvalSet:
    return EvalSet.model_validate_json(SET_PATH.read_text(encoding="utf-8"))


def test_loads_as_adk_type(eval_set: EvalSet) -> None:
    assert eval_set.eval_set_id == "weather_agent_v1"
    assert len(eval_set.eval_cases) == 10


def test_eval_ids_unique(eval_set: EvalSet) -> None:
    ids = [c.eval_id for c in eval_set.eval_cases]
    assert len(ids) == len(set(ids))


def test_tool_calls_reference_real_tool(eval_set: EvalSet) -> None:
    for case in eval_set.eval_cases:
        for use in case.conversation[0].intermediate_data.tool_uses:
            assert use.name == "get_weather", f"{case.eval_id}: 存在しないツール {use.name}"
            assert "city" in use.args, f"{case.eval_id}: city が無い"


def test_expected_answers_match_tool_output(eval_set: EvalSet) -> None:
    """正解の文が、ツールが実際に返す値と整合しているか。

    ツールの固定データを変えたのに評価セットを直し忘れると、
    正しい実装が落ちる。ここで先に落とす。
    """
    checked = 0
    for case in eval_set.eval_cases:
        inv = case.conversation[0]
        answer = inv.final_response.parts[0].text
        for use in inv.intermediate_data.tool_uses:
            city = use.args["city"]
            result = get_weather(city)
            if result["status"] != "success":
                continue
            # 気温の数字が正解の文に入っているか。
            temp = result["report"].split("気温 ")[1].split(" 度")[0]
            assert temp in answer, f"{case.eval_id}: {city} の気温 {temp} が答えに無い"
            checked += 1
    assert checked >= 5, f"照合できたのが {checked} 件"


def test_unknown_city_expects_tool_call(eval_set: EvalSet) -> None:
    """未登録の都市でもツールを呼ぶことを期待する。

    呼ばずに答えたら、ツールではなく知識から答えている。
    出力は正しく見えても手順が違う。
    """
    case = next(c for c in eval_set.eval_cases if c.eval_id == "unknown_city")
    uses = case.conversation[0].intermediate_data.tool_uses
    assert len(uses) == 1
    city = uses[0].args["city"]
    assert city not in _WEATHER, "未登録のはずの都市が登録されている"
    assert get_weather(city)["status"] == "error"


def test_out_of_scope_expects_no_tool_call(eval_set: EvalSet) -> None:
    for eval_id in ("out_of_scope", "vague_user", "prompt_injection"):
        case = next(c for c in eval_set.eval_cases if c.eval_id == eval_id)
        uses = case.conversation[0].intermediate_data.tool_uses
        assert uses == [], f"{eval_id}: ツールを呼ぶ期待になっている"


def test_covers_personas_and_attacks(eval_set: EvalSet) -> None:
    """ペルソナと攻撃の観点が含まれているか。

    正常系だけの評価セットは、通ることしか確かめない。
    """
    ids = {c.eval_id for c in eval_set.eval_cases}
    for required in ("hurried_user", "vague_user", "prompt_injection"):
        assert required in ids, f"{required} が無い"


def test_city_args_are_lowercase_ascii(eval_set: EvalSet) -> None:
    """ツールの docstring が指定する形式に沿っているか。"""
    for case in eval_set.eval_cases:
        for use in case.conversation[0].intermediate_data.tool_uses:
            city = use.args["city"]
            assert city == city.lower(), f"{case.eval_id}: {city} が小文字でない"
            assert city.isascii(), f"{case.eval_id}: {city} が英字でない"
