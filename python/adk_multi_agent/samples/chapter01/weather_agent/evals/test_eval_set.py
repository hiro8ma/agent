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

from samples.chapter01.weather_agent.agent import root_agent
from samples.chapter01.weather_agent.tools import (
    _SIGHTSEEING,
    _WEATHER,
    get_sightseeing,
    get_weather,
    normalize,
)

TOOLS = {"get_weather": get_weather, "get_sightseeing": get_sightseeing}

SET_PATH = Path(__file__).parent / "weather_agent_v1.evalset.json"


def _instruction_text() -> str:
    """動的 Instruction を State なしで組み立てて中身を見る。"""
    from google.adk.agents.readonly_context import ReadonlyContext

    class _Empty:
        state: dict = {}

    return root_agent.instruction(_Empty())


@pytest.fixture(scope="module")
def eval_set() -> EvalSet:
    return EvalSet.model_validate_json(SET_PATH.read_text(encoding="utf-8"))


def test_loads_as_adk_type(eval_set: EvalSet) -> None:
    assert eval_set.eval_set_id == "weather_agent_v1"
    assert len(eval_set.eval_cases) == 15


def test_eval_ids_unique(eval_set: EvalSet) -> None:
    ids = [c.eval_id for c in eval_set.eval_cases]
    assert len(ids) == len(set(ids))


def test_tool_calls_reference_real_tool(eval_set: EvalSet) -> None:
    for case in eval_set.eval_cases:
        for use in case.conversation[0].intermediate_data.tool_uses:
            assert use.name in TOOLS, f"{case.eval_id}: 存在しないツール {use.name}"
            assert "city" in use.args, f"{case.eval_id}: city が無い"

        declared = {t.__name__ for t in root_agent.tools}
        for use in case.conversation[0].intermediate_data.tool_uses:
            assert use.name in declared, f"{case.eval_id}: エージェントに未登録の {use.name}"


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
            result = TOOLS[use.name](city)
            if result["status"] != "success":
                continue
            if use.name == "get_weather":
                assert str(result["temperature"]) in answer, (
                    f"{case.eval_id}: {city} の気温 {result['temperature']} が答えに無い"
                )
            else:
                assert result["spots"][0] in answer, (
                    f"{case.eval_id}: {city} のスポット {result['spots'][0]} が答えに無い"
                )
            checked += 1
    assert checked >= 5, f"照合できたのが {checked} 件"


def test_unknown_city_expects_tool_call(eval_set: EvalSet) -> None:
    """未登録の都市でもツールを呼ぶことを期待する。

    呼ばずに答えたら、ツールではなく知識から答えている。
    出力は正しく見えても手順が違う。
    """
    for eval_id, data, fn in (
        ("unknown_city", _WEATHER, get_weather),
        ("sightseeing_unknown", _SIGHTSEEING, get_sightseeing),
    ):
        case = next(c for c in eval_set.eval_cases if c.eval_id == eval_id)
        uses = case.conversation[0].intermediate_data.tool_uses
        assert len(uses) == 1
        city = normalize(uses[0].args["city"])
        assert city not in data, f"{eval_id}: 未登録のはずの都市が登録されている"
        assert fn(city)["status"] == "error"


def test_out_of_scope_expects_no_tool_call(eval_set: EvalSet) -> None:
    for eval_id in ("out_of_scope", "vague_user", "prompt_injection", "credential_request"):
        case = next(c for c in eval_set.eval_cases if c.eval_id == eval_id)
        uses = case.conversation[0].intermediate_data.tool_uses
        assert uses == [], f"{eval_id}: ツールを呼ぶ期待になっている"


def test_instruction_covers_every_expected_behavior(eval_set: EvalSet) -> None:
    """評価セットが期待する振る舞いを Instruction が指示しているか。

    Instruction に無い振る舞いを評価セットが期待すると、実装ではなく
    仕様の欠落で落ちる。落ちた側を直しても直らない。
    """
    instruction = _instruction_text()
    required = {
        "vague_user": "聞き返",
        "out_of_scope": "無関係",
        "prompt_injection": "上書き",
        "no_extrapolation": "予報",
        "unknown_city": "登録されていない",
        "trip_suggestion": "組み合わせ",
        "credential_request": "無関係",
    }
    ids = {c.eval_id for c in eval_set.eval_cases}
    for eval_id, phrase in required.items():
        if eval_id in ids:
            assert phrase in instruction, (
                f"{eval_id} を期待しているが Instruction に {phrase!r} が無い"
            )


def test_memory_case_declares_prior_state(eval_set: EvalSet) -> None:
    """記憶を使うケースが、前提の State を宣言しているか。

    State を宣言せずに「前回の都市」を期待すると、実装ではなく
    評価セットの前提不足で落ちる。
    """
    from samples.chapter01.weather_agent.tools import LAST_CITY_KEY

    case = next(c for c in eval_set.eval_cases if c.eval_id == "recall_last_city")
    assert case.session_input is not None, "前提の State が無い"
    assert LAST_CITY_KEY in case.session_input.state, f"{LAST_CITY_KEY} が無い"

    # 動的 Instruction がその State を読むことを確かめる。
    city = case.session_input.state[LAST_CITY_KEY]
    from samples.chapter01.weather_agent.agent import build_instruction

    class _WithState:
        state = {LAST_CITY_KEY: city}

    assert city in build_instruction(_WithState()), (
        "Instruction が直近の都市を差し込んでいない"
    )


def test_covers_personas_and_attacks(eval_set: EvalSet) -> None:
    """ペルソナと攻撃の観点が含まれているか。

    正常系だけの評価セットは、通ることしか確かめない。
    """
    ids = {c.eval_id for c in eval_set.eval_cases}
    for required in ("hurried_user", "vague_user", "prompt_injection"):
        assert required in ids, f"{required} が無い"


def test_city_args_are_accepted_by_tool(eval_set: EvalSet) -> None:
    """期待する引数を、ツールが実際に受け付けるか。

    形式を決め打ちで検査すると、ツール側が別名を受けるようになっても
    評価セットだけが古い形式に縛られる。ツールに渡して確かめる。
    """
    for case in eval_set.eval_cases:
        for use in case.conversation[0].intermediate_data.tool_uses:
            city = use.args["city"]
            fn = TOOLS[use.name]
            if case.eval_id.endswith("unknown") or case.eval_id == "unknown_city":
                assert fn(city)["status"] == "error"
                continue
            assert fn(city)["status"] == "success", (
                f"{case.eval_id}: {use.name} が {city!r} を受け付けない"
            )
