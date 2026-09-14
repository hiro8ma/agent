"""v2.2.0 のコンテキスト型の形を固定する。

教材は 3 種類（InvocationContext / ReadonlyContext / Context）に整理し、
CallbackContext と ToolContext は Context の別名だと書いている。
別名なのか別の型なのかで、書ける範囲が変わる。実物で確かめる。

読み取り専用が型だけの約束なのか、実行時にも止まるのかも見る。
"""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest
from google.adk import Context
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import ToolContext


def _fake_invocation(state: dict) -> SimpleNamespace:
    """ReadonlyContext.state が触るのは session.state だけ。"""
    return SimpleNamespace(session=SimpleNamespace(state=state))


def test_callback_and_tool_context_are_aliases_of_context():
    """別名なので、型で書き込みの可否を分けてはいない。"""
    assert CallbackContext is Context
    assert ToolContext is Context


def test_context_extends_readonly_context():
    """Context は ReadonlyContext を継承し、書き込み側の口を足した形。"""
    assert issubclass(Context, ReadonlyContext)
    for name in ("state", "agent_name", "invocation_id"):
        assert hasattr(ReadonlyContext, name)
    # ツールとコールバックで使う口は Context 側にだけある
    for name in ("function_call_id", "actions", "save_artifact", "load_artifact"):
        assert hasattr(Context, name)
        assert not hasattr(ReadonlyContext, name)


def test_readonly_state_blocks_writes_at_runtime():
    """型の約束だけでなく、実行時にも書けない。

    state は MappingProxyType を返すので、代入は TypeError になる。
    動的 instruction を関数で書くとき、State を壊す書き方が通らない。
    """
    ctx = ReadonlyContext(_fake_invocation({"city": "tokyo"}))
    assert isinstance(ctx.state, MappingProxyType)
    assert ctx.state["city"] == "tokyo"
    with pytest.raises(TypeError):
        ctx.state["city"] = "osaka"


def test_readonly_state_is_a_view_not_a_copy():
    """複製ではなく参照。後で書かれた値は instruction 生成側からも見える。"""
    session_state: dict = {"city": "tokyo"}
    ctx = ReadonlyContext(_fake_invocation(session_state))
    session_state["city"] = "osaka"
    assert ctx.state["city"] == "osaka"
