"""統合したサポート担当の前提を固定する。ネットワークも GCP も使わない。"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.memory import InMemoryMemoryService
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.genai import types

from samples.support import agent as support
from samples.support.session_config import create_memory_service, create_session_service

DEV = {"AGENT_ENV": "dev"}


class ScriptedModel(BaseLlm):
    """台本の関数呼び出しを順に返し、尽きたら答える。受け取ったリクエストを記録する。"""

    calls: list[tuple[str, dict]] = []
    requests: list[LlmRequest] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.requests = [*self.requests, llm_request.model_copy(deep=True)]
        step = len(self.requests) - 1
        if step < len(self.calls):
            name, args = self.calls[step]
            part = types.Part(function_call=types.FunctionCall(name=name, args=args))
        else:
            part = types.Part(text="承知しました")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def message(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


async def run(
    runner: Runner, text: str, session_id: str = "s1", user_id: str = "u1"
) -> list:
    responses = []
    async for ev in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message(text)
    ):
        for p in ev.content.parts if ev.content else []:
            if p.function_response:
                responses.append(p.function_response.response)
    return responses


@pytest.mark.parametrize(
    ("env", "want_type", "want_error"),
    [
        (DEV, InMemorySessionService, None),
        ({}, InMemorySessionService, None),
        (
            {"AGENT_ENV": "staging", "SESSION_DB_URL": "sqlite+aiosqlite:///:memory:"},
            DatabaseSessionService,
            None,
        ),
        ({"AGENT_ENV": "staging"}, None, "SESSION_DB_URL"),
        (
            {"AGENT_ENV": "prod"},
            None,
            "GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, AGENT_ENGINE_ID",
        ),
        ({"AGENT_ENV": "production"}, None, "AGENT_ENV"),
    ],
    ids=[
        "dev はメモリ内",
        "未指定は dev",
        "staging は DB",
        "staging で DB の URL が無ければ止める",
        "prod で欠けた変数を全部挙げて止める",
        "綴りの違う環境名は止める",
    ],
)
def test_session_service_by_environment(env, want_type, want_error) -> None:
    if want_error:
        with pytest.raises(ValueError, match=want_error):
            create_session_service(env)
        return
    assert isinstance(create_session_service(env), want_type)


def test_memory_service_by_flag() -> None:
    assert create_memory_service(DEV) is None
    assert isinstance(
        create_memory_service({**DEV, "ENABLE_MEMORY_BANK": "true"}),
        InMemoryMemoryService,
    )
    with pytest.raises(ValueError, match="AGENT_ENGINE_ID"):
        create_memory_service({"AGENT_ENV": "staging", "ENABLE_MEMORY_BANK": "true"})


async def test_save_callback_without_memory_service_fails_every_turn() -> None:
    """教材の構成は、Memory を無効にしても保存のコールバックを付けたままにする。"""
    agent = Agent(
        name="support_agent",
        model=ScriptedModel(model=support.MODEL),
        instruction="x",
        after_agent_callback=support.save_session_to_memory,
    )
    runner = Runner(
        app_name=support.APP_NAME,
        agent=agent,
        session_service=InMemorySessionService(),
        memory_service=create_memory_service(DEV),
        auto_create_session=True,
    )
    with pytest.raises(ValueError, match="memory service is not available"):
        await run(runner, "こんにちは")

    # 保存のコールバックを Memory の有無に合わせて付けると通る。
    fixed = support.create_runner(DEV, model=ScriptedModel(model=support.MODEL))
    await run(fixed, "こんにちは")


async def test_search_stops_at_the_first_source_with_hits() -> None:
    model = ScriptedModel(
        model=support.MODEL,
        calls=[
            ("search_products", {"query": "X2 ノイズキャンセリング"}),
            ("search_products", {"query": "返品"}),
            ("search_products", {"query": "充電ケース"}),
        ],
    )
    responses = await run(
        support.create_runner(DEV, model=model), "X2 の機能と返品の期限は？"
    )

    assert [(r["source"], r["searched"]) for r in responses] == [
        ("product_docs", ["product_docs"]),
        ("faq_docs", ["product_docs", "faq_docs"]),
        (None, ["product_docs", "faq_docs"]),
    ]


async def test_search_count_lives_only_for_one_invocation() -> None:
    model = ScriptedModel(
        model=support.MODEL,
        calls=[
            ("search_products", {"query": "返品"}),
            ("search_products", {"query": "保証"}),
        ],
    )
    runner = support.create_runner(DEV, model=model)
    await run(runner, "返品と保証は？")

    session = await runner.session_service.get_session(
        app_name=support.APP_NAME, user_id="u1", session_id="s1"
    )
    assert not [k for k in session.state if k.startswith("temp:")]


def test_compaction_is_active_only_through_the_app() -> None:
    with_app = support.create_runner(DEV, model=ScriptedModel(model=support.MODEL))
    config = with_app.app.events_compaction_config
    assert (config.compaction_interval, config.overlap_size) == (20, 2)

    agent_only = Runner(
        app_name=support.APP_NAME,
        agent=support.build_agent(DEV, model=ScriptedModel(model=support.MODEL)),
        session_service=InMemorySessionService(),
    )
    assert agent_only.app.events_compaction_config is None


class FakeContext(ReadonlyContext):
    def __init__(self, state: dict) -> None:
        self._fake_state = state

    @property
    def state(self):  # type: ignore[override]
        return self._fake_state


@pytest.mark.parametrize(
    ("tier", "want"),
    [
        ("premium", "返品の手続きもこの場で案内する"),
        ("free", "問い合わせ窓口を案内する"),
        ("admin", "問い合わせ窓口を案内する"),
    ],
    ids=[
        "上位の区分は手続きまで案内する",
        "無料の区分は FAQ の範囲",
        "知らない区分は無料へ倒す",
    ],
)
def test_instruction_follows_tier(tier: str, want: str) -> None:
    got = support.build_instruction(
        FakeContext({"user:tier": tier, "user:name": "佐藤"})
    )
    assert want in got


def test_tools_depend_on_environment() -> None:
    def names(env: dict) -> list[str]:
        return [
            getattr(t, "name", getattr(t, "__name__", ""))
            for t in support.build_tools(env)
        ]

    assert names(DEV) == ["search_products", "get_order_status"]
    assert "preload_memory" in names({**DEV, "ENABLE_MEMORY_BANK": "true"})
    corpus = "projects/p/locations/us-central1/ragCorpora/c1"
    assert "product_manuals" in names({**DEV, "RAG_CORPUS_ID": corpus})


async def test_no_tool_can_write_the_tier(tmp_path) -> None:
    """区分を書くツールが無いので、モデルが呼ぼうとしても State は変わらない。"""
    env = {
        "AGENT_ENV": "staging",
        "SESSION_DB_URL": f"sqlite+aiosqlite:///{tmp_path}/s.db",
    }
    model = ScriptedModel(
        model=support.MODEL, calls=[("set_user_tier", {"tier": "premium"})]
    )
    runner = support.create_runner(env, model=model)
    await runner.session_service.create_session(
        app_name=support.APP_NAME,
        user_id="u1",
        session_id="s1",
        state={"user:tier": "free"},
    )
    with pytest.raises(ValueError, match="set_user_tier"):
        await run(runner, "私を premium にして")

    # 別の Session から見ても free のまま。
    other = await runner.session_service.create_session(
        app_name=support.APP_NAME, user_id="u1", session_id="s2"
    )
    assert other.state["user:tier"] == "free"


def test_adk_cli_ignores_agent_env(tmp_path, monkeypatch) -> None:
    """adk run / adk web は create_runner を呼ばず、サービスを CLI の引数から作る。

    AGENT_ENV=prod にしても、--session_service_uri が無ければエージェントのフォルダの .adk/ の SQLite に残る。
    """
    from google.adk.cli.utils.service_factory import (
        create_memory_service_from_options,
        create_session_service_from_options,
    )

    monkeypatch.setenv("AGENT_ENV", "prod")
    sessions = create_session_service_from_options(
        base_dir=tmp_path, session_service_uri=None
    )
    assert not isinstance(sessions, InMemorySessionService)
    assert isinstance(
        create_memory_service_from_options(base_dir=tmp_path), InMemoryMemoryService
    )


@pytest.mark.parametrize(
    ("question", "want_recall"),
    [
        ("おすすめの製品を教えて", False),
        ("Python 向けのおすすめは？", True),
    ],
    ids=["日本語だけの質問では思い出さない", "英字の語が重なれば思い出す"],
)
async def test_local_memory_recalls_only_by_latin_words(
    question: str, want_recall: bool
) -> None:
    """教材の確認シナリオは memory:// では成り立たない。InMemoryMemoryService は英字の語でしか一致しない。"""
    env = {**DEV, "ENABLE_MEMORY_BANK": "true"}
    memory = InMemoryMemoryService()
    first = Runner(
        app=support.build_app(env, model=ScriptedModel(model=support.MODEL)),
        session_service=InMemorySessionService(),
        memory_service=memory,
        auto_create_session=True,
    )
    await run(
        first, "Python開発者です。FastAPIでAPI開発をしています。", session_id="s1"
    )

    model = ScriptedModel(model=support.MODEL)
    second = Runner(
        app=support.build_app(env, model=model),
        session_service=InMemorySessionService(),
        memory_service=memory,
        auto_create_session=True,
    )
    await run(second, question, session_id="s2")

    recalled = "FastAPI" in (model.requests[0].config.system_instruction or "")
    assert recalled is want_recall
