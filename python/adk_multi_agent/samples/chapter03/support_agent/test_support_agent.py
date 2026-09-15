"""統合した構成の配線を、API キー無しで固定する。

見るのは 6 つ。

    Instruction   ティアで対応方針が変わり、State を書けない
    注入          進行中の注文が要求の末尾へ入る
    上限          回数を超えるとモデルを呼ばずに返し、注入も走らない
    出力          禁止語を含む応答が安全な文へ差し替わる
    権限          viewer では取り消しが実行されず、editor では実行される
    整形          検索結果が 5 件に縮み、落とした件数が残る
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from samples.chapter03.support_agent.agent import root_agent
from samples.chapter03.support_agent.callbacks import (
    ACTIVE_ORDERS_KEY,
    MODEL_CALL_KEY,
    ROLE_KEY,
    TIER_KEY,
    build_instruction,
)


class FakeReadonlyContext:
    """build_instruction が触るのは state だけ。"""

    def __init__(self, state: dict) -> None:
        self.state = state


class ScriptedModel(BaseLlm):
    """台本の順に応答を返し、受け取った要求を残すモデル。"""

    turns: list[types.Content] = []
    calls: int = 0
    seen_texts: list[str] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.seen_texts = self.seen_texts + [
            part.text
            for content in llm_request.contents or []
            for part in content.parts or []
            if part.text
        ]
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


def call(name: str, args: dict) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
    )


def text(body: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=body)])


async def run(model: ScriptedModel, message: str, state: dict | None = None) -> dict:
    """台本のモデルを差し込んで 1 往復させる。"""
    app = f"ch03_{id(model)}"
    original = root_agent.model
    root_agent.model = model
    try:
        runner = InMemoryRunner(agent=root_agent, app_name=app)
        await runner.session_service.create_session(
            app_name=app, user_id="u1", session_id="s1", state=state or {}
        )
        tool_results: list[dict] = []
        final = ""
        async for event in runner.run_async(
            user_id="u1",
            session_id="s1",
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        ):
            for part in (
                event.content.parts if event.content and event.content.parts else []
            ):
                if part.function_response is not None:
                    tool_results.append(dict(part.function_response.response or {}))
                if part.text:
                    final = part.text
        session = await runner.session_service.get_session(
            app_name=app, user_id="u1", session_id="s1"
        )
        artifacts = await runner.artifact_service.list_artifact_keys(
            app_name=app, user_id="u1", session_id="s1"
        )
        return {
            "tools": tool_results,
            "final": final,
            "state": dict(session.state),
            "artifacts": artifacts,
        }
    finally:
        root_agent.model = original


def test_instruction_changes_with_the_tier():
    free = build_instruction(FakeReadonlyContext({TIER_KEY: "free"}))
    premium = build_instruction(FakeReadonlyContext({TIER_KEY: "premium"}))
    assert "3 文以内" in free
    assert "優先窓口" in premium
    # 未知のティアは最小の方針へ倒す
    assert "3 文以内" in build_instruction(FakeReadonlyContext({TIER_KEY: "unknown"}))
    assert "3 文以内" in build_instruction(FakeReadonlyContext({}))


def test_agent_wires_all_four_callbacks():
    """4 種すべてがリストで配線されている。合成関数は使っていない。"""
    for callbacks in (
        root_agent.before_model_callback,
        root_agent.after_model_callback,
        root_agent.before_tool_callback,
        root_agent.after_tool_callback,
    ):
        assert isinstance(callbacks, list) and callbacks
    assert len(root_agent.before_model_callback) == 2


async def test_active_orders_are_injected_at_the_end():
    model = ScriptedModel(model="scripted", turns=[text("確認しました")])
    result = await run(
        model, "最近の注文について教えて", state={ACTIVE_ORDERS_KEY: ["ORD-12345"]}
    )
    assert any("進行中の注文: ORD-12345" in t for t in model.seen_texts)
    assert result["final"] == "確認しました"


async def test_rate_limit_skips_the_model_and_the_injection():
    """上限を超えた要求では、モデルも呼ばれず注入も走らない。"""
    model = ScriptedModel(model="scripted", turns=[text("本来の応答")])
    result = await run(
        model,
        "注文の状況を教えて",
        state={MODEL_CALL_KEY: 10, ACTIVE_ORDERS_KEY: ["ORD-12345"]},
    )
    assert result["final"] == "混み合っています。時間をおいて試してください。"
    assert model.calls == 0
    assert model.seen_texts == []


async def test_banned_word_is_replaced():
    model = ScriptedModel(
        model="scripted", turns=[text("クレジットカード番号は 4111-1111 です")]
    )
    result = await run(model, "教えて")
    assert result["final"] == "申し訳ありません。その内容はお伝えできません。"


async def test_viewer_cannot_cancel_but_editor_can():
    script = [
        call("cancel_order", {"order_id": "ORD-12345", "reason": "サイズ違い"}),
        text("完了"),
    ]

    viewer = ScriptedModel(model="scripted", turns=list(script))
    blocked = await run(viewer, "キャンセルして", state={ROLE_KEY: "viewer"})
    assert blocked["tools"][0]["status"] == "error"
    assert "editor" in blocked["tools"][0]["message"]
    assert blocked["artifacts"] == []

    editor = ScriptedModel(model="scripted", turns=list(script))
    allowed = await run(editor, "キャンセルして", state={ROLE_KEY: "editor"})
    assert allowed["tools"][0]["status"] == "cancelled"
    # 取り消しの記録はツールが成果物として残す
    assert allowed["tools"][0]["artifact_version"] == 0
    assert allowed["artifacts"] == ["cancel_ORD-12345.txt"]


async def test_shipped_order_is_refused_by_the_tool():
    """対照。権限があっても、発送済みはツール側が断る。"""
    model = ScriptedModel(
        model="scripted",
        turns=[
            call("cancel_order", {"order_id": "ORD-99999", "reason": "不要"}),
            text("案内"),
        ],
    )
    result = await run(model, "キャンセルして", state={ROLE_KEY: "admin"})
    assert "発送済み" in result["tools"][0]["error"]


async def test_search_results_are_trimmed_to_five():
    model = ScriptedModel(
        model="scripted",
        turns=[
            call("search_products", {"query": "シャツ", "category": ""}),
            text("5 件出しました"),
        ],
    )
    result = await run(model, "シャツある？")
    payload = result["tools"][0]
    assert len(payload["results"]) == 5
    assert payload["note"] == "全 10 件のうち上位 5 件"


@pytest.mark.parametrize(
    "name", ["list_skills", "load_skill", "load_skill_resource", "run_skill_script"]
)
async def test_skill_tools_are_available_next_to_function_tools(name: str):
    """スキルの操作ツールと関数ツールが同じ tools に並ぶ。"""
    tools = await root_agent.canonical_tools()
    names = {tool.name for tool in tools}
    assert name in names
    assert {"get_order_status", "cancel_order", "search_products"} <= names
