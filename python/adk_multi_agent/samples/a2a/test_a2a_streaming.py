"""A2A のストリーミングと再接続、ADK のオーケストレーションとの併用の前提を確かめる。モデルは台本で置き換える。"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory, create_text_message_object
from a2a.client.errors import A2AClientJSONRPCError
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskStatusUpdateEvent,
)
from google.adk import Agent
from google.adk.a2a.converters.request_converter import (
    convert_a2a_request_to_agent_run_request,
)
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.executor.config import A2aAgentExecutorConfig
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.run_config import StreamingMode
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

BASE = "http://localhost:8001"
CHUNKS = ["7月の", "経費は", "12件です"]
STREAMING_CARD = AgentCard(
    name="expense",
    description="経費",
    url=BASE,
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    default_input_modes=["text"],
    default_output_modes=["text"],
    skills=[
        AgentSkill(id="expense", name="経費", description="経費", tags=["expense"])
    ],
)


class Chunks(BaseLlm):
    """ストリーミングで呼ばれたら 3 つに分けて返し、最後に全文を返す。"""

    streamed: list[bool] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del llm_request
        self.streamed = [*self.streamed, stream]
        if stream:
            for chunk in CHUNKS:
                content = types.Content(role="model", parts=[types.Part(text=chunk)])
                yield LlmResponse(content=content, partial=True)
        content = types.Content(role="model", parts=[types.Part(text="".join(CHUNKS))])
        yield LlmResponse(content=content)


def streaming_request(request, part_converter):
    """既定の変換は RunConfig にストリーミングの指定を入れないので、SSE を足す。"""
    run = convert_a2a_request_to_agent_run_request(request, part_converter)
    run.run_config.streaming_mode = StreamingMode.SSE
    return run


def with_sse(runner) -> A2aAgentExecutor:
    return A2aAgentExecutor(
        runner=runner,
        config=A2aAgentExecutorConfig(request_converter=streaming_request),
    )


async def collect(card: AgentCard | None, executor_factory=None) -> tuple[list, Chunks]:
    model = Chunks(model="gemini-3.8-flash")
    app = to_a2a(
        Agent(name="expense", model=model, instruction="x"),
        host="localhost",
        port=8001,
        agent_card=card,
        agent_executor_factory=executor_factory,
    )
    events: list = []
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as hc,
    ):
        client = await ClientFactory.connect(
            BASE, client_config=ClientConfig(httpx_client=hc, streaming=True)
        )
        message = create_text_message_object(content="7月の経費を集計して")
        async for event in client.send_message(message):
            task, update = event if isinstance(event, tuple) else (None, event)
            if (
                isinstance(update, TaskStatusUpdateEvent)
                and update.status.state == "working"
            ):
                msg = update.status.message
                text = "".join(
                    getattr(p.root, "text", "") for p in (msg.parts if msg else [])
                )
                events.append(("working", text))
            elif isinstance(update, TaskArtifactUpdateEvent):
                events.append(
                    ("artifact", "".join(p.root.text for p in update.artifact.parts))
                )
            elif update is None and task is not None:
                events.append(("task", task.status.state.value))
    return events, model


async def test_auto_card_does_not_stream() -> None:
    """Agent Card を省くと capabilities が空で、完了の 1 件しか届かない。"""
    events, model = await collect(card=None)
    assert events == [("task", "completed")]
    assert model.streamed == [False]


async def test_streaming_card_streams_status_but_not_text() -> None:
    """streaming: true を宣言しても、モデルはストリーミングで呼ばれず、途中の出力は届かない。全文だけが届く。"""
    events, model = await collect(card=STREAMING_CARD)
    assert model.streamed == [False]
    assert [e[1] for e in events if e[0] == "working" and e[1]] == [
        "7月の経費は12件です"
    ]
    assert events[-1] == ("artifact", "7月の経費は12件です")


async def test_sse_mode_streams_partial_text_as_working_status() -> None:
    """実行器の変換で SSE を指定すると、途中の出力が working の状態のメッセージとして順に届く。全文は 2 回届く。"""
    events, model = await collect(card=STREAMING_CARD, executor_factory=with_sse)
    assert model.streamed == [True]
    texts = [e[1] for e in events if e[0] == "working" and e[1]]
    assert texts == [*CHUNKS, "7月の経費は12件です"]
    assert events[-1] == ("artifact", "7月の経費は12件です")


async def test_resubscribe_to_a_finished_task_is_an_error() -> None:
    """切れている間にタスクが終わっていると、再接続はエラーになる。先に get_task で状態を見る。"""
    app = to_a2a(
        Agent(name="expense", model=Chunks(model="gemini-3.8-flash"), instruction="x"),
        host="localhost",
        port=8001,
        agent_card=STREAMING_CARD,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as hc,
    ):
        client = await ClientFactory.connect(
            BASE, client_config=ClientConfig(httpx_client=hc, streaming=True)
        )
        task_id = None
        async for event in client.send_message(
            create_text_message_object(content="集計して")
        ):
            if isinstance(event, tuple):
                task_id = event[0].id
        with pytest.raises(A2AClientJSONRPCError, match="terminal state"):
            async for _ in client.resubscribe(TaskIdParams(id=task_id)):
                pass


async def test_connection_failure_is_not_a_connection_error() -> None:
    """接続できないときの例外は組み込みの ConnectionError の子ではない。except ConnectionError では捕まらない。"""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as dead:
        client = await ClientFactory.connect(
            STREAMING_CARD,
            client_config=ClientConfig(httpx_client=dead, streaming=True),
        )
        with pytest.raises(Exception) as caught:
            async for _ in client.resubscribe(TaskIdParams(id="t1")):
                pass
    assert not isinstance(caught.value, ConnectionError)


@pytest.mark.parametrize(
    ("instruction", "fails"),
    [
        ("内部処理の結果 {formatted_data} を確認し", True),
        ("内部処理の結果 {formatted_data?} を確認し", False),
    ],
    ids=["State に無い変数は KeyError で止まる", "? を付ければ空で続く"],
)
async def test_instruction_variable_needs_the_internal_step(
    instruction: str, fails: bool
) -> None:
    """併用の例の external_coordinator は、内部の処理と順に並べないと formatted_data が無い。"""
    agent = Agent(
        name="external_coordinator",
        model=Chunks(model="gemini-3.8-flash"),
        instruction=instruction,
    )
    runner = InMemoryRunner(agent=agent, app_name="x")
    await runner.session_service.create_session(
        app_name="x", user_id="u", session_id="s"
    )
    message = types.Content(role="user", parts=[types.Part(text="送って")])

    async def run() -> None:
        async for _ in runner.run_async(
            user_id="u", session_id="s", new_message=message
        ):
            pass

    if fails:
        with pytest.raises(KeyError, match="formatted_data"):
            await run()
    else:
        await run()
