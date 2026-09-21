"""RAG と Memory を 1 つのエージェントで併用するときの前提を固定する。

教材の「ツール分離型」と「コールバック統合型」を、組み立てられたリクエストで確かめる。
ネットワークも GCP も使わない。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.memory import InMemoryMemoryService
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.load_memory_tool import load_memory
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.retrieval import VertexAiRagRetrieval
from google.genai import types

MODEL = "gemini-3.8-flash"


def rag(name: str, corpus: str) -> VertexAiRagRetrieval:
    return VertexAiRagRetrieval(
        name=name,
        description=f"{name} を検索する",
        rag_corpora=[f"projects/p/locations/us-central1/ragCorpora/{corpus}"],
        similarity_top_k=3,
    )


def lookup_order(order_id: str) -> dict:
    """注文の状態を返す。"""
    return {"order_id": order_id, "status": "shipped"}


class ScriptedModel(BaseLlm):
    """1 回目はツールを呼び、2 回目で答える。受け取ったリクエストを記録する。"""

    requests: list[LlmRequest] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.requests = [*self.requests, llm_request.model_copy(deep=True)]
        if len(self.requests) == 1:
            part = types.Part(
                function_call=types.FunctionCall(
                    name="lookup_order", args={"order_id": "A1"}
                )
            )
        else:
            part = types.Part(text="発送済みです")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


class PlainModel(BaseLlm):
    """ツールを呼ばずに答える。過去の会話を作るのに使う。"""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request, stream
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="承知しました")])
        )


def tool_kinds(request: LlmRequest) -> dict[str, list[str]]:
    tools = request.config.tools or []
    return {
        "retrieval": [
            t.retrieval.vertex_rag_store.rag_resources[0].rag_corpus
            if t.retrieval.vertex_rag_store.rag_resources
            else "corpora"
            for t in tools
            if t.retrieval
        ],
        "google_search": ["google_search" for t in tools if t.google_search],
        "functions": sorted(
            d.name
            for t in tools
            if t.function_declarations
            for d in t.function_declarations
        ),
    }


async def run_once(agent: Agent, memory: InMemoryMemoryService | None = None) -> None:
    runner = Runner(
        app_name="support",
        agent=agent,
        session_service=InMemorySessionService(),
        memory_service=memory or InMemoryMemoryService(),
        auto_create_session=True,
    )
    message = types.Content(role="user", parts=[types.Part(text="注文 A1 の状態は？")])
    async for _ in runner.run_async(user_id="u1", session_id="s1", new_message=message):
        pass


def test_append_instructions_rejects_a_plain_string() -> None:
    """教材のコールバックは文字列を 1 つ渡すが、受け付けるのは list[str] か Content だけ。"""
    request = LlmRequest(model=MODEL, config=types.GenerateContentConfig())
    with pytest.raises(TypeError):
        request.append_instructions("[関連FAQ]\n返品は 30 日以内")  # type: ignore[arg-type]

    request.append_instructions(["[関連FAQ]\n返品は 30 日以内"])
    assert "返品は 30 日以内" in request.config.system_instruction


async def test_before_model_callback_runs_on_every_model_call() -> None:
    """ツールを 1 回呼ぶ質問でモデルは 2 回呼ばれ、事前注入の検索も 2 回走る。"""
    searches: list[str] = []

    async def inject(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        searches.append("faq")
        llm_request.append_instructions(["[関連FAQ]\n発送後のキャンセルは返品扱い"])

    model = ScriptedModel(model=MODEL)
    agent = Agent(
        name="support",
        model=model,
        instruction="サポート担当",
        tools=[lookup_order],
        before_model_callback=inject,
    )
    await run_once(agent)

    assert len(model.requests) == 2
    assert len(searches) == 2
    # 注入は毎回組み立て直すリクエストに入るので、重ならない。
    for request in model.requests:
        assert request.config.system_instruction.count("発送後のキャンセル") == 1


async def test_rag_with_load_memory_mixes_built_in_and_function() -> None:
    """ツール分離型で Memory を必要時だけ引かせると、組み込みの検索と関数が同じリクエストに並ぶ。

    ADK は google_search と VertexAiSearchTool にだけ、この組み合わせを避ける置き換えを持つ。
    VertexAiRagRetrieval には無い。
    """
    model = ScriptedModel(model=MODEL)
    agent = Agent(
        name="support",
        model=model,
        instruction="サポート担当",
        tools=[
            rag("product_docs", "c1"),
            rag("policy_docs", "c2"),
            load_memory,
            lookup_order,
        ],
    )
    await run_once(agent)

    kinds = tool_kinds(model.requests[0])
    assert len(kinds["retrieval"]) == 2
    assert kinds["functions"] == ["load_memory", "lookup_order"]


async def test_preload_memory_is_not_a_tool_the_model_chooses() -> None:
    """PreloadMemoryTool は関数として宣言されず、instruction に過去の会話を差し込む。"""
    memory = InMemoryMemoryService()
    seed = Runner(
        app_name="support",
        agent=Agent(name="support", model=PlainModel(model=MODEL), instruction="x"),
        session_service=InMemorySessionService(),
        memory_service=memory,
        auto_create_session=True,
    )
    # InMemoryMemoryService は英字の語でしか一致しないので、英字を含む発話にする。
    past = types.Content(
        role="user", parts=[types.Part(text="order A1 was late last time")]
    )
    async for _ in seed.run_async(user_id="u1", session_id="old", new_message=past):
        pass
    old = await seed.session_service.get_session(
        app_name="support", user_id="u1", session_id="old"
    )
    await memory.add_session_to_memory(old)

    model = ScriptedModel(model=MODEL)
    agent = Agent(
        name="support",
        model=model,
        instruction="サポート担当",
        tools=[PreloadMemoryTool(), lookup_order],
    )
    runner = Runner(
        app_name="support",
        agent=agent,
        session_service=InMemorySessionService(),
        memory_service=memory,
        auto_create_session=True,
    )
    message = types.Content(role="user", parts=[types.Part(text="order A1 status?")])
    async for _ in runner.run_async(
        user_id="u1", session_id="new", new_message=message
    ):
        pass

    first = model.requests[0]
    assert tool_kinds(first)["functions"] == ["lookup_order"]
    assert "order A1 was late last time" in first.config.system_instruction


@pytest.mark.parametrize(
    ("search_tool", "want_built_in", "want_functions"),
    [
        (google_search, ["google_search"], ["lookup_order"]),
        (
            GoogleSearchTool(bypass_multi_tools_limit=True),
            [],
            ["google_search_agent", "lookup_order"],
        ),
    ],
    ids=[
        "既定では組み込みの検索と関数が並ぶ",
        "bypass を付けると検索をサブエージェントに包む",
    ],
)
async def test_google_search_with_other_tools(
    search_tool: GoogleSearchTool, want_built_in: list[str], want_functions: list[str]
) -> None:
    model = ScriptedModel(model=MODEL)
    agent = Agent(
        name="support",
        model=model,
        instruction="サポート担当",
        tools=[search_tool, lookup_order],
    )
    await run_once(agent)

    kinds = tool_kinds(model.requests[0])
    assert kinds["google_search"] == want_built_in
    assert kinds["functions"] == want_functions
