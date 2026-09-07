"""落とし穴 2 つを検査で固定する。

どちらもエラーを出さないため、検査が無ければ本番で気づく。
"""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
from google.adk import Agent
from google.adk.agents import LoopAgent, ParallelAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from samples.orchestration.pipeline import MAX_ROUNDS, build_loop, build_nested

# 暴走を止める上限。これに当たったら「止まらない」と判定する。
_EMERGENCY_STOP = 40


class NeverApproves(BaseLlm):
    """承認を返さないモデル。上限が効かなければ止まらない。"""

    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        if self.calls > _EMERGENCY_STOP:
            raise RuntimeError("emergency stop")
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="まだ直して")])
        )


def _agent(name: str, text: str, key: str, model: BaseLlm | None = None) -> Agent:
    class Fixed(BaseLlm):
        out: str = ""

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=self.out)])
            )

    return Agent(
        name=name,
        model=model or Fixed(model="fixed", out=text),
        instruction="返す",
        output_key=key,
    )


async def _state(root, app: str) -> dict:
    runner = InMemoryRunner(agent=root, app_name=app)
    await runner.session_service.create_session(
        app_name=app, user_id="u", session_id="s"
    )
    async for _ in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(role="user", parts=[types.Part(text="go")]),
    ):
        pass
    session = await runner.session_service.get_session(
        app_name=app, user_id="u", session_id="s"
    )
    return dict(session.state)


async def test_loop_without_max_iterations_does_not_stop():
    """max_iterations 未設定では止まらないことを見る。

    教材は「必須パラメータではない」と書くだけで、
    設定しなかったときに何が起きるかを書いていない。
    既定は None で、None は無制限になる。
    """
    model = NeverApproves(model="never")
    loop = LoopAgent(
        name="unbounded",
        sub_agents=[
            _agent("gen", "", "code", model),
            _agent("rev", "", "verdict", model),
        ],
    )
    with pytest.raises(Exception):
        await _state(loop, "unbounded")
    assert model.calls > _EMERGENCY_STOP, f"止まった。呼び出し {model.calls} 回"


async def test_bounded_loop_stops():
    """上限を渡せば止まることを見る。対照。"""
    model = NeverApproves(model="never")
    loop = build_loop(
        _agent("gen", "", "code", model),
        _agent("rev", "", "verdict", model),
    )
    await _state(loop, "bounded")
    # 子 2 つ × MAX_ROUNDS 周
    assert model.calls == 2 * MAX_ROUNDS, f"呼び出し {model.calls} 回"


async def test_colliding_output_keys_lose_results_silently():
    """同じ output_key へ並列に書くと、1 つだけ残ることを見る。

    エラーも警告も出ない。3 回分の費用を払って 1 つしか残らない。
    """
    collide = ParallelAgent(
        name="collide",
        sub_agents=[
            _agent("a", "AAA", "result"),
            _agent("b", "BBB", "result"),
            _agent("c", "CCC", "result"),
        ],
    )
    state = await _state(collide, "collide")
    assert list(state) == ["result"], state
    assert len(state) == 1, f"衝突していない: {state}"


async def test_distinct_output_keys_keep_every_result():
    """鍵を分ければ全部残ることを見る。対照。"""
    ok = ParallelAgent(
        name="ok",
        sub_agents=[
            _agent("a2", "AAA", "flight"),
            _agent("b2", "BBB", "hotel"),
            _agent("c2", "CCC", "activity"),
        ],
    )
    state = await _state(ok, "ok")
    assert state == {"flight": "AAA", "hotel": "BBB", "activity": "CCC"}, state


def test_build_nested_rejects_duplicate_keys():
    """組み立て時に鍵の重複を弾くことを見る。

    実行してから 1 つしか残らないと気づくのでは遅い。
    """
    with pytest.raises(ValueError, match="output_key"):
        build_nested(
            [_agent("a", "A", "same"), _agent("b", "B", "same")],
            _agent("p", "P", "plan"),
            _agent("r", "R", "report"),
        )


async def test_nested_shares_state_across_levels():
    """入れ子でも state が共通であることを見る。"""
    nested = build_nested(
        [_agent("f", "FLIGHT", "flight"), _agent("h", "HOTEL", "hotel")],
        _agent("plan", "PLAN", "plan"),
        _agent("report", "REPORT", "report"),
    )
    state = await _state(nested, "nested")
    for key in ("flight", "hotel", "plan", "report"):
        assert key in state, f"{key} が state に無い: {state}"


def test_template_workflows_are_all_deprecated():
    """3 つとも非推奨であることを見る。

    教材は「単純な定型フローには Template Workflow を使い」と勧めるが、
    v2.2.0 では 3 つとも非推奨で Workflow を指している。
    将来外れたら、この検査が落ちて気づける。
    """
    import warnings

    from google.adk.agents import LoopAgent, ParallelAgent, SequentialAgent

    for cls in (SequentialAgent, ParallelAgent, LoopAgent):
        # 子は毎回作り直す。1 つのエージェントは親を 1 つしか持てない。
        child = _agent(f"child_{cls.__name__}", "X", "x")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cls(name=f"probe_{cls.__name__}", sub_agents=[child])
        messages = " ".join(str(w.message) for w in caught)
        assert "deprecated" in messages, f"{cls.__name__} に非推奨の警告が無い"
        assert "Workflow" in messages, f"{cls.__name__} の代替が示されていない"


def test_an_agent_cannot_have_two_parents():
    """1 つのエージェントインスタンスを 2 つの親へ入れられないことを見る。

    同じ子を使い回すと ValidationError になる。
    並列と順次で同じエージェントを共有したくなるが、できない。
    """
    from google.adk.agents import ParallelAgent, SequentialAgent

    child = _agent("shared", "X", "x")
    SequentialAgent(name="first_parent", sub_agents=[child])
    with pytest.raises(Exception, match="already has a parent"):
        ParallelAgent(name="second_parent", sub_agents=[child])
