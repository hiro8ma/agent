"""A2A のクライアント側で、リモートのエージェントの入力待ち（input-required）に答える経路を確かめる。

サーバーは to_a2a をメモリ内で立て、モデルは台本で置き換える。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

import httpx
import pytest
from a2a.client import Client, ClientConfig, ClientFactory, create_text_message_object
from a2a.types import Message, Role, Task, TaskState
from google.adk import Agent
from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, request_input
from google.genai import types

BASE = "http://localhost:8001"
CARD = f"{BASE}/.well-known/agent-card.json"


class Expense(BaseLlm):
    """ランチ代は金額を聞き返す。聞き返しへの答えを受ければ登録し、それ以外は新しい依頼として受ける。"""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        last = llm_request.contents[-1]
        answer = next(
            (p.function_response for p in last.parts or [] if p.function_response), None
        )
        text = "".join(p.text or "" for p in last.parts or [])
        if answer is not None:
            part = types.Part(text=f"登録しました: {answer.response['answer']}")
        elif "ランチ" in text:
            call = types.FunctionCall(
                name="adk_request_input", args={"message": "金額を教えてください"}
            )
            part = types.Part(function_call=call)
        else:
            part = types.Part(text=f"新しい依頼として受け付けました: {text}")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


@pytest.fixture
async def http() -> AsyncGenerator[tuple[httpx.AsyncClient, list[str]], None]:
    agent = Agent(
        name="expense_agent",
        model=Expense(model="gemini-3.8-flash"),
        instruction="x",
        tools=[request_input],
    )
    app = to_a2a(agent, host="localhost", port=8001)
    card_gets: list[str] = []

    async def count(request: httpx.Request) -> None:
        if request.url.path.endswith("agent-card.json"):
            card_gets.append(str(request.url))

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url=BASE,
            event_hooks={"request": [count]},
        ) as client,
    ):
        yield client, card_gets


async def send(client: Client, message: Message) -> Task:
    last = None
    async for event in client.send_message(message):
        last = event[0] if isinstance(event, tuple) else event
    assert isinstance(last, Task)
    return last


def texts(task: Task) -> list[str]:
    parts = [p for a in task.artifacts or [] for p in a.parts]
    return [p.root.text for p in parts if hasattr(p.root, "text")]


def pending_call(task: Task) -> dict:
    for part in task.status.message.parts:
        if (getattr(part.root, "metadata", None) or {}).get(
            "adk_type"
        ) == "function_call":
            return part.root.data
    raise AssertionError("input_required のタスクに FunctionCall が無い")


@pytest.mark.parametrize(
    ("reply", "same_task", "want"),
    [
        ("function_response", True, "登録しました: 1200円"),
        ("text_with_ids", True, "新しい依頼として受け付けました: 1200円"),
        ("text_without_ids", False, "新しい依頼として受け付けました: 1200円"),
    ],
    ids=[
        "FunctionResponse で答えると聞き返しに答えられる",
        "テキストで答えると完了になるが聞き返しには答えていない",
        "ID を付けないと別のタスクになる",
    ],
)
async def test_low_level_client_answers_input_required(
    http: tuple[httpx.AsyncClient, list[str]], reply: str, same_task: bool, want: str
) -> None:
    hc, _ = http
    client = await ClientFactory.connect(
        BASE, client_config=ClientConfig(httpx_client=hc, streaming=False)
    )
    task = await send(
        client, create_text_message_object(content="昨日のランチ代を登録して")
    )
    assert task.status.state == TaskState.input_required

    if reply == "function_response":
        call = pending_call(task)
        genai_part = types.Part(
            function_response=types.FunctionResponse(
                id=call["id"], name=call["name"], response={"answer": "1200円"}
            )
        )
        message = Message(
            role=Role.user,
            parts=[convert_genai_part_to_a2a_part(genai_part)],
            message_id=str(uuid.uuid4()),
            task_id=task.id,
            context_id=task.context_id,
        )
    else:
        message = create_text_message_object(content="1200円")
        if reply == "text_with_ids":
            message.task_id, message.context_id = task.id, task.context_id

    followup = await send(client, message)
    assert followup.status.state == TaskState.completed
    assert (followup.id == task.id) is same_task
    assert texts(followup) == [want]


async def run_turn(runner: InMemoryRunner, message: types.Content) -> list[types.Part]:
    parts: list[types.Part] = []
    async for ev in runner.run_async(user_id="u", session_id="s", new_message=message):
        parts += ev.content.parts if ev.content else []
    return parts


async def test_remote_agent_surfaces_the_question_and_accepts_a_function_response(
    http: tuple[httpx.AsyncClient, list[str]],
) -> None:
    """RemoteA2aAgent を直接使うと、聞き返しは長時間のツールの呼び出しとして届き、同じ ID の FunctionResponse で答えられる。"""
    hc, card_gets = http
    remote = RemoteA2aAgent(name="expense_agent", agent_card=CARD, httpx_client=hc)
    runner = InMemoryRunner(agent=remote, app_name="office")
    await runner.session_service.create_session(
        app_name="office", user_id="u", session_id="s"
    )

    first = await run_turn(
        runner,
        types.Content(role="user", parts=[types.Part(text="昨日のランチ代を登録して")]),
    )
    [call] = [p.function_call for p in first if p.function_call]
    assert call.name == "adk_request_input"

    answer = types.Part(
        function_response=types.FunctionResponse(
            id=call.id, name=call.name, response={"answer": "1200円"}
        )
    )
    second = await run_turn(runner, types.Content(role="user", parts=[answer]))
    assert [p.text for p in second if p.text] == ["登録しました: 1200円"]
    # Agent Card は最初の 1 回だけ取りに行き、以後はキャッシュを使う。
    assert len(card_gets) == 1


class Orchestrator(BaseLlm):
    """受けた発話をそのまま expense_agent に渡し、ツールの結果を記録する。"""

    results: list[dict] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        last = llm_request.contents[-1]
        result = next(
            (p.function_response for p in last.parts or [] if p.function_response), None
        )
        if result is not None:
            self.results = [*self.results, result.response]
            part = types.Part(text="終わりました")
        else:
            text = "".join(p.text or "" for p in last.parts or [])
            part = types.Part(
                function_call=types.FunctionCall(
                    name="expense_agent", args={"request": text}
                )
            )
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


async def test_agent_tool_loses_the_question_and_the_task(
    http: tuple[httpx.AsyncClient, list[str]],
) -> None:
    """AgentTool で包むと、聞き返しは親に空の結果として届き、次の呼び出しは新しい依頼になる。"""
    hc, _ = http
    remote = RemoteA2aAgent(
        name="expense_agent", description="経費", agent_card=CARD, httpx_client=hc
    )
    model = Orchestrator(model="gemini-3.8-flash")
    orchestrator = Agent(
        name="office", model=model, instruction="x", tools=[AgentTool(agent=remote)]
    )
    runner = InMemoryRunner(agent=orchestrator, app_name="office")
    await runner.session_service.create_session(
        app_name="office", user_id="u", session_id="s"
    )

    for text in ["昨日のランチ代を登録して", "1200円"]:
        await run_turn(
            runner, types.Content(role="user", parts=[types.Part(text=text)])
        )

    assert model.results == [
        {"result": ""},
        {"result": "新しい依頼として受け付けました: 1200円"},
    ]
