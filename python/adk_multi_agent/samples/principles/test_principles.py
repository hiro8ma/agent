"""設計原則の教材のコード例が前提にしていることを、台本のモデルで確かめる。モデルは呼ばない。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import AsyncGenerator

import httpx
from google.adk import Agent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def reply(*parts: types.Part) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=list(parts)))


def call(name: str, **args) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


class Once(BaseLlm):
    """最初に決めた呼び出しを返し、ツールの結果を受けたら終わる。"""

    first: list = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        if llm_request.contents[-1].parts[-1].function_response:
            yield reply(types.Part(text="終わりました"))
        else:
            yield reply(*self.first)


async def run(agent: Agent, state_delta: dict | None = None) -> InMemoryRunner:
    runner = InMemoryRunner(agent=agent, app_name="p")
    await runner.session_service.create_session(
        app_name="p", user_id="u", session_id="s"
    )
    message = types.Content(role="user", parts=[types.Part(text="お願い")])
    async for _ in runner.run_async(
        user_id="u", session_id="s", new_message=message, state_delta=state_delta
    ):
        pass
    return runner


# --- 原則 2 最小権限: State に置いた役割は呼び出し元が書ける ---


def update_price(product_id: str, price: int) -> dict:
    """価格を更新する。"""
    return {"status": "updated", "product_id": product_id, "price": price}


def check_tool_permission(tool: BaseTool, args: dict, tool_context: ToolContext):
    """教材の権限の確認。役割を State から読む。"""
    del args
    if tool.name == "update_price" and tool_context.state.get(
        "user_role", "viewer"
    ) not in (
        "admin",
        "editor",
    ):
        return {"error": f"権限不足: {tool.name}"}
    return None


async def test_role_in_state_can_be_set_by_the_caller() -> None:
    """ランナーの state_delta（REST の /run でも受け付ける）で、呼び出し元が自分を admin にできる。"""
    results: list[dict] = []

    def record(tool, args, tool_context, tool_response):
        del tool, args, tool_context
        results.append(tool_response)

    def agent() -> Agent:
        return Agent(
            name="admin_agent",
            model=Once(
                model="m", first=[call("update_price", product_id="A", price=1)]
            ),
            instruction="x",
            tools=[update_price],
            before_tool_callback=check_tool_permission,
            after_tool_callback=record,
        )

    await run(agent())
    await run(agent(), state_delta={"user_role": "admin"})
    assert results[0] == {"error": "権限不足: update_price"}
    assert results[1]["status"] == "updated"


# --- 原則 3 冪等性: 引数のハッシュを冪等キーにすると、正当な 2 回目を弾く ---


class Orders:
    def __init__(self, gap: float = 0.0) -> None:
        self.rows: dict[str, str] = {}
        self.gap = gap

    def create_book(self, customer_id: str, product_id: str, quantity: int) -> dict:
        """教材の create_order。冪等キーを引数から作り、探してから入れる。"""
        key = hashlib.sha256(
            json.dumps(
                {
                    "customer_id": customer_id,
                    "product_id": product_id,
                    "quantity": quantity,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        if key in self.rows:
            return {"status": "already_exists", "order_id": self.rows[key]}
        time.sleep(self.gap)
        order_id = f"O-{len(self.rows) + 1}"
        self.rows[key] = order_id
        return {"status": "created", "order_id": order_id}


def test_args_hash_rejects_a_legitimate_repeat_order() -> None:
    """同じ顧客が翌週に同じ商品を同じ数だけ注文すると、新しい注文ではなく前の注文が返る。"""
    orders = Orders()
    first = orders.create_book("C1", "P1", 1)
    next_week = orders.create_book("C1", "P1", 1)
    assert first["status"] == "created"
    assert next_week == {"status": "already_exists", "order_id": first["order_id"]}


def test_check_then_insert_races() -> None:
    """探してから入れる間に別の呼び出しが来ると、同じキーで 2 件作る。"""
    orders = Orders(gap=0.05)
    results: list[dict] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(orders.create_book("C1", "P1", 1))
        )
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r["status"] for r in results] == ["created", "created"]


# --- 原則 4 可観測性: 開始時刻を State に置く書き方 ---


async def slow(seconds: float) -> dict:
    """指定の秒数だけ待つ。"""
    await asyncio.sleep(seconds)
    return {"slept": seconds}


async def test_start_time_in_state_is_per_call_but_persisted() -> None:
    """1 ターンの 2 つの呼び出しは並行に走るが、State の書き込みは呼び出しごとに分かれ、上書きし合わない。
    その代わり、開始時刻はセッションの State に残る。"""
    measured: dict[float, float] = {}

    def before(tool, args, tool_context):
        del tool, args
        tool_context.state["_tool_start_time"] = time.monotonic()

    def after(tool, args, tool_context, tool_response):
        del tool, tool_response
        measured[args["seconds"]] = (
            time.monotonic() - tool_context.state["_tool_start_time"]
        )

    agent = Agent(
        name="observable_agent",
        model=Once(
            model="m", first=[call("slow", seconds=0.3), call("slow", seconds=0.05)]
        ),
        instruction="x",
        tools=[slow],
        before_tool_callback=before,
        after_tool_callback=after,
    )
    started = time.monotonic()
    runner = await run(agent)
    # 並行に走るので、全体は 0.35 秒ではなく 0.3 秒ほどで終わる。
    assert time.monotonic() - started < 0.35
    assert 0.3 <= measured[0.3] < 0.35
    assert 0.05 <= measured[0.05] < 0.1
    session = await runner.session_service.get_session(
        app_name="p", user_id="u", session_id="s"
    )
    assert "_tool_start_time" in session.state


# --- 原則 9 フェイルセーフ: HTTP クライアントのタイムアウトは組み込みの TimeoutError ではない ---


def test_httpx_timeout_is_not_builtin_timeout_error() -> None:
    """教材の except TimeoutError では、httpx のタイムアウトを捕まえられない。"""
    assert not issubclass(httpx.TimeoutException, TimeoutError)
