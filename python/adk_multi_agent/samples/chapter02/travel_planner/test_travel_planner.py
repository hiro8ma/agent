"""旅行プランナーの配線を API キー無しで固定する検査。

見るのは 4 つ。

    パイプラインの形      並列 3 本 → 日程 → 予算の順序と、output_key の一意性
    State の受け渡し      動的 instruction が 3 つの鍵を読み、欠けても落ちない
    ツールの約束          戻り値は常に list、例外を投げずに error を返す
    スキーマの約束        全フィールドに description、入れ子は 3 層まで

台本どおりに答えるモデルを差し込み、5 エージェントを 1 往復させて
State に鍵が積まれ、最後が TravelPlan として検証されるところまで見る。
"""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncGenerator

import pytest
from google.adk import Agent
from google.adk.agents import ParallelAgent, SequentialAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel

from samples.chapter02.travel_planner.agent import root_agent
from samples.chapter02.travel_planner.agents import (
    REPORT_KEY,
    RESTAURANT_KEY,
    SCHEDULE_KEY,
    SPOT_KEY,
    TRANSPORT_KEY,
    budget_reporter,
    research_phase,
    schedule_planner,
)
from samples.chapter02.travel_planner.agents.planner import planner_instruction
from samples.chapter02.travel_planner.agents.reporter import reporter_instruction
from samples.chapter02.travel_planner.schemas import TravelPlan
from samples.chapter02.travel_planner.tools import (
    search_restaurants,
    search_tourist_spots,
    search_transport,
    spot_search_tool,
)

MODEL_NAME = "gemini-3.5-flash"


class FakeContext:
    """ReadonlyContext の代わり。instruction が読むのは state だけ。"""

    def __init__(self, state: dict) -> None:
        self.state = state


def test_pipeline_shape():
    """並列 3 本 → 日程 → 予算の順に並んでいるか。"""
    assert isinstance(root_agent, SequentialAgent)
    assert [a.name for a in root_agent.sub_agents] == [
        "research_phase",
        "schedule_planner",
        "budget_reporter",
    ]
    assert isinstance(research_phase, ParallelAgent)
    assert [a.name for a in research_phase.sub_agents] == [
        "spot_researcher",
        "restaurant_researcher",
        "transport_researcher",
    ]


def test_every_llm_agent_declares_the_book_model():
    """モデルの指定漏れがあると、既定のモデルで静かに動いてしまう。"""
    for agent in [*research_phase.sub_agents, schedule_planner, budget_reporter]:
        assert agent.model == MODEL_NAME, agent.name


def test_output_keys_do_not_collide():
    """並列の子は State を共有する。鍵が重なると後勝ちで消える。"""
    keys = [a.output_key for a in research_phase.sub_agents]
    keys += [schedule_planner.output_key, budget_reporter.output_key]
    assert keys == [SPOT_KEY, RESTAURANT_KEY, TRANSPORT_KEY, SCHEDULE_KEY, REPORT_KEY]
    assert len(set(keys)) == len(keys)


def test_template_workflow_agents_are_deprecated():
    """v2.2.0 では生成時に警告が出る。消えたらこの検査で気づく。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        child = Agent(name="child", model=MODEL_NAME, instruction="x")
        SequentialAgent(
            name="seq", sub_agents=[ParallelAgent(name="par", sub_agents=[child])]
        )
    messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(messages) == 2, messages
    for m in messages:
        assert "Please use Workflow instead." in m


def test_planner_instruction_reads_all_three_keys():
    """3 つの調査結果が instruction に入るか。"""
    ctx = FakeContext(
        {
            SPOT_KEY: "金閣寺と伏見稲荷",
            RESTAURANT_KEY: "権太呂",
            TRANSPORT_KEY: "新幹線 13320 円",
        }
    )
    text = planner_instruction(ctx)
    for want in ("金閣寺と伏見稲荷", "権太呂", "新幹線 13320 円"):
        assert want in text


def test_planner_instruction_survives_missing_keys():
    """並列の 1 本が落ちても、日程の生成まで進める。"""
    text = planner_instruction(FakeContext({}))
    assert "観光スポットの調査結果なし" in text
    assert "レストランの調査結果なし" in text
    assert "交通手段の調査結果なし" in text


def test_reporter_instruction_reads_schedule():
    """予算は日程から計算する。日程が無ければその旨が入る。"""
    text = reporter_instruction(FakeContext({SCHEDULE_KEY: "1 日目: 金閣寺"}))
    assert "1 日目: 金閣寺" in text
    assert "日程表なし" in reporter_instruction(FakeContext({}))


def test_dynamic_instruction_does_not_substitute_braces():
    """instruction を関数にすると波括弧の置換は行われない。

    `{spot_research}` と書いても State の値には変わらないので、
    ctx.state から自分で読む必要がある。
    """
    text = planner_instruction(FakeContext({SPOT_KEY: "{restaurant_research}"}))
    assert "{restaurant_research}" in text


@pytest.mark.parametrize(
    "call",
    [
        lambda: search_tourist_spots("京都", ["歴史"]),
        lambda: search_tourist_spots("該当なし市", []),
        lambda: search_restaurants("京都", "和食", 5000),
        lambda: search_restaurants("京都", "和食", 1),
        lambda: search_transport("東京", "京都", "2026-10-03"),
        lambda: search_transport("東京", "該当なし市", "2026-10-03"),
    ],
)
def test_tools_always_return_a_list(call):
    """成功でも失敗でも戻り値の型を変えない。宣言は型ヒントから作られる。"""
    result = call()
    assert isinstance(result, list)
    assert all(isinstance(item, dict) for item in result)


def test_tools_report_errors_without_raising():
    """例外を投げるとモデルは何が起きたか分からず、同じ呼び出しを繰り返す。"""
    assert search_tourist_spots("該当なし市", [])[0]["error"]
    assert search_restaurants("京都", "和食", 1)[0]["error"]
    assert search_transport("東京", "該当なし市", "2026-10-03")[0]["error"]


def test_tool_filtering_narrows_results():
    """interests と予算で絞れているか。"""
    history = search_tourist_spots("京都", ["歴史"])
    assert {s["category"] for s in history} == {"歴史"}
    cheap = search_restaurants("京都", "", 3000)
    assert all(s["budget_per_person_yen"] <= 3000 for s in cheap)


def test_tool_docstrings_are_usable_as_declarations():
    """1 行目が説明、Args に例がある状態を保つ。"""
    for func in (search_tourist_spots, search_restaurants, search_transport):
        doc = func.__doc__ or ""
        first = doc.strip().splitlines()[0]
        assert first.endswith("。"), func.__name__
        assert "Args:" in doc, func.__name__
        assert "例:" in doc, func.__name__


def _depth(model: type[BaseModel], seen: tuple[type, ...] = ()) -> int:
    """Pydantic モデルの入れ子の深さ。"""
    best = 1
    for field in model.model_fields.values():
        for arg in (field.annotation, *getattr(field.annotation, "__args__", ())):
            if isinstance(arg, type) and issubclass(arg, BaseModel) and arg not in seen:
                best = max(best, 1 + _depth(arg, (*seen, model)))
    return best


def test_schema_fields_all_have_description():
    """description はモデルへ渡る仕様書。空だと形が揺れる。"""
    stack: list[type[BaseModel]] = [TravelPlan]
    checked: set[type[BaseModel]] = set()
    while stack:
        model = stack.pop()
        if model in checked:
            continue
        checked.add(model)
        for name, field in model.model_fields.items():
            assert field.description, f"{model.__name__}.{name}"
            for arg in (field.annotation, *getattr(field.annotation, "__args__", ())):
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    stack.append(arg)
    assert len(checked) >= 5


def test_schema_nesting_is_within_three_levels():
    assert _depth(TravelPlan) <= 3


def test_output_schema_and_tools_can_coexist():
    """v2.2.0 では併用が許される。旧版の「併用不可」から変わった点。"""
    agent = Agent(
        name="reporter_with_tools",
        model=MODEL_NAME,
        instruction="x",
        tools=[spot_search_tool],
        output_schema=TravelPlan,
        output_key="probe",
    )
    assert agent.output_schema is TravelPlan
    assert agent.tools


class ScriptedModel(BaseLlm):
    """台本の順に応答を返すモデル。"""

    turns: list[types.Content] = []
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        i = self.calls
        self.calls = i + 1
        if i >= len(self.turns):
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="台本の終わり")]
                )
            )
            return
        yield LlmResponse(content=self.turns[i])


def _call(name: str, args: dict) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
    )


def _text(s: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=s)])


_PLAN = {
    "destination": "京都",
    "duration_days": 2,
    "schedule": [
        {
            "day": 1,
            "spots": [
                {
                    "name": "金閣寺",
                    "category": "歴史",
                    "duration_hours": 1.5,
                    "entry_fee_yen": 500,
                }
            ],
            "restaurants": [
                {
                    "name": "権太呂",
                    "cuisine": "和食",
                    "meal": "昼食",
                    "budget_per_person_yen": 2500,
                }
            ],
            "notes": "市バスで移動",
        }
    ],
    "transport": [{"mode": "新幹線", "duration_minutes": 135, "price_yen": 13320}],
    "budget": {
        "transport_yen": 26640,
        "food_yen": 2500,
        "activity_yen": 500,
        "total_yen": 29640,
    },
    "highlights": ["金閣寺", "権太呂の昼食"],
    "tips": ["市バスの一日券が安い"],
}


@pytest.mark.asyncio
async def test_pipeline_fills_state_in_order_with_scripted_models():
    """5 エージェントを 1 往復させ、State に鍵が積まれるところまで見る。"""
    scripts = {
        "spot_researcher": [
            _call("search_tourist_spots", {"city": "京都", "interests": ["歴史"]}),
            _text("金閣寺と伏見稲荷大社がおすすめです。"),
        ],
        "restaurant_researcher": [
            _call(
                "search_restaurants",
                {"city": "京都", "cuisine": "和食", "budget_per_person": 5000},
            ),
            _text("権太呂は 2500 円ほどです。"),
        ],
        "transport_researcher": [
            _call(
                "search_transport",
                {"origin": "東京", "destination": "京都", "date": "2026-10-03"},
            ),
            _text("新幹線が 135 分 13320 円です。"),
        ],
        "schedule_planner": [_text("1 日目 午前 金閣寺、昼 権太呂")],
        "budget_reporter": [_text(json.dumps(_PLAN, ensure_ascii=False))],
    }
    agents = {
        a.name: a
        for a in [*research_phase.sub_agents, schedule_planner, budget_reporter]
    }
    originals = {name: agent.model for name, agent in agents.items()}
    for name, agent in agents.items():
        agent.model = ScriptedModel(model="scripted", turns=scripts[name])
    try:
        runner = InMemoryRunner(agent=root_agent, app_name="ch02_plan")
        await runner.session_service.create_session(
            app_name="ch02_plan", user_id="u1", session_id="s1"
        )
        tool_calls: list[str] = []
        async for event in runner.run_async(
            user_id="u1",
            session_id="s1",
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part(
                        text="東京から京都へ 2 泊。歴史が好き。食事は 5000 円まで"
                    )
                ],
            ),
        ):
            for part in (
                event.content.parts if event.content and event.content.parts else []
            ):
                if part.function_response is not None:
                    tool_calls.append(part.function_response.name)

        assert sorted(tool_calls) == [
            "search_restaurants",
            "search_tourist_spots",
            "search_transport",
        ], tool_calls

        session = await runner.session_service.get_session(
            app_name="ch02_plan", user_id="u1", session_id="s1"
        )
        state = dict(session.state)
        for key in (SPOT_KEY, RESTAURANT_KEY, TRANSPORT_KEY, SCHEDULE_KEY, REPORT_KEY):
            assert key in state, sorted(state)

        # 最終出力は TravelPlan として読める形か
        report = state[REPORT_KEY]
        plan = TravelPlan.model_validate(
            json.loads(report) if isinstance(report, str) else report
        )
        assert plan.destination == "京都"
        assert plan.budget.total_yen == (
            plan.budget.transport_yen + plan.budget.food_yen + plan.budget.activity_yen
        )
    finally:
        for name, agent in agents.items():
            agent.model = originals[name]
