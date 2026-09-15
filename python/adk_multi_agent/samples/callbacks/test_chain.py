"""コールバックの並べ方と打ち切りの規則を固定する。

見るのは 4 つ。

    リスト渡し   ADK が順に呼ぶ。合成関数を自前で書く必要はない
    打ち切り     戻り値が真になった時点で止まる。None でないかではない
    空 dict      偽なので止まらない。ツールも飛ばない
    差し替え     before_tool はツールの結果に、after_tool は結果の加工に使う
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk import Agent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from samples.callbacks.chain import (
    MODEL_CALL_KEY,
    ROLE_KEY,
    authorize_tools,
    inject_note,
    limit_model_calls,
    log_requests,
    redact_response,
    shrink_tool_result,
)

MODEL_NAME = "scripted"

# ツールが実際に走ったかを見るための記録。テストごとに空にする。
CALLS: list[str] = []


def cancel_order(order_id: str) -> dict:
    """注文を取り消す。

    Args:
        order_id: 注文 ID。例: A-1001
    """
    CALLS.append(f"cancel_order:{order_id}")
    return {"status": "ok", "order_id": order_id}


def search_products(query: str) -> dict:
    """商品を検索する。

    Args:
        query: 検索語。例: シャツ
    """
    CALLS.append(f"search_products:{query}")
    return {"results": [{"name": f"item{i}"} for i in range(10)]}


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


def call(name: str, args: dict) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
    )


def text(body: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=body)])


async def run(agent: Agent, message: str, state: dict | None = None) -> dict:
    """1 往復させ、ツール結果と最終応答と State を返す。"""
    app = f"cb_{id(agent)}"
    runner = InMemoryRunner(agent=agent, app_name=app)
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
    return {"tools": tool_results, "final": final, "state": dict(session.state)}


@pytest.fixture(autouse=True)
def _clear_calls():
    CALLS.clear()
    yield
    CALLS.clear()


async def test_callback_list_runs_in_order():
    """リストで渡せば順に走る。合成関数は要らない。"""
    seen: list[str] = []
    agent = Agent(
        name="ordered",
        model=ScriptedModel(model=MODEL_NAME, turns=[text("はい")]),
        instruction="x",
        before_model_callback=[log_requests(seen), inject_note("補足: 在庫は前日時点")],
    )
    result = await run(agent, "在庫は？")
    assert len(seen) == 1
    assert result["final"] == "はい"


async def test_chain_stops_at_the_first_truthy_return():
    """途中で LlmResponse を返すと、そこで止まりモデルも呼ばれない。"""
    after: list[str] = []
    model = ScriptedModel(model=MODEL_NAME, turns=[text("本来の応答")])
    agent = Agent(
        name="limited",
        model=model,
        instruction="x",
        before_model_callback=[
            limit_model_calls(0, message="上限です"),
            log_requests(after),
        ],
    )
    result = await run(agent, "何か")
    assert result["final"] == "上限です"
    assert model.calls == 0  # モデルは呼ばれない
    assert after == []  # 後続のコールバックも走らない
    assert result["state"][MODEL_CALL_KEY] == 1  # 数えた分は State に残る


async def test_after_model_callback_replaces_the_response():
    """禁止語を含む応答を差し替える。"""
    agent = Agent(
        name="redacted",
        model=ScriptedModel(model=MODEL_NAME, turns=[text("パスワードは 1234 です")]),
        instruction="x",
        after_model_callback=[redact_response(["パスワード"], "お伝えできません")],
    )
    result = await run(agent, "教えて")
    assert result["final"] == "お伝えできません"


async def test_before_tool_callback_blocks_by_role():
    """viewer では書き込み系のツールが実行されない。"""
    agent = Agent(
        name="guarded",
        model=ScriptedModel(
            model=MODEL_NAME,
            turns=[call("cancel_order", {"order_id": "A-1001"}), text("できません")],
        ),
        instruction="x",
        tools=[cancel_order],
        before_tool_callback=[authorize_tools(["cancel_order"])],
    )
    result = await run(agent, "取り消して")
    assert CALLS == []  # ツール本体は走っていない
    assert result["tools"][0]["status"] == "error"


async def test_admin_role_passes_the_guard():
    """対照。admin なら通る。"""
    agent = Agent(
        name="allowed",
        model=ScriptedModel(
            model=MODEL_NAME,
            turns=[
                call("cancel_order", {"order_id": "A-1001"}),
                text("取り消しました"),
            ],
        ),
        instruction="x",
        tools=[cancel_order],
        before_tool_callback=[authorize_tools(["cancel_order"])],
    )
    result = await run(agent, "取り消して", state={ROLE_KEY: "admin"})
    assert CALLS == ["cancel_order:A-1001"]
    assert result["tools"][0]["status"] == "ok"


async def test_empty_dict_skips_the_tool_but_not_the_chain():
    """空の dict はツールを飛ばすが、鎖は止めない。

    判定している条件が 2 か所で違う。

        鎖を止めるか      `if function_response:`          真のときだけ止まる
        ツールを呼ぶか    `if function_response is None:`   None のときだけ呼ぶ

    そのため、通したつもりで空の dict を返すとツールが黙って飛ぶ。
    通すときは None を返す。
    """

    def returns_empty(tool, args, tool_context) -> dict:
        return {}

    def returns_marker(tool, args, tool_context) -> dict:
        return {"status": "error", "message": "2 本目が返した値"}

    agent = Agent(
        name="empty_guard",
        model=ScriptedModel(
            model=MODEL_NAME,
            turns=[call("cancel_order", {"order_id": "A-1002"}), text("完了")],
        ),
        instruction="x",
        tools=[cancel_order],
        before_tool_callback=[returns_empty, returns_marker],
    )
    result = await run(agent, "取り消して")
    assert CALLS == []  # ツールは飛ぶ
    assert result["tools"][0]["message"] == "2 本目が返した値"  # 鎖は進む


async def test_returning_none_lets_the_tool_run():
    """対照。None を返せばツールは通常どおり走る。"""

    def passes(tool, args, tool_context) -> dict | None:
        return None

    agent = Agent(
        name="pass_through",
        model=ScriptedModel(
            model=MODEL_NAME,
            turns=[call("cancel_order", {"order_id": "A-1003"}), text("完了")],
        ),
        instruction="x",
        tools=[cancel_order],
        before_tool_callback=[passes],
    )
    result = await run(agent, "取り消して")
    assert CALLS == ["cancel_order:A-1003"]
    assert result["tools"][0]["status"] == "ok"


async def test_after_tool_callback_shrinks_the_result():
    """大きな結果を上位 3 件に縮め、落とした件数を書き添える。"""
    agent = Agent(
        name="shrunk",
        model=ScriptedModel(
            model=MODEL_NAME,
            turns=[
                call("search_products", {"query": "シャツ"}),
                text("3 件出しました"),
            ],
        ),
        instruction="x",
        tools=[search_products],
        after_tool_callback=[
            shrink_tool_result("results", keep=3, tools=["search_products"])
        ],
    )
    result = await run(agent, "シャツある？")
    payload = result["tools"][0]
    assert len(payload["results"]) == 3
    assert payload["note"] == "全 10 件のうち上位 3 件"
