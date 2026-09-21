"""Hub-Spoke の経費精算を、台本のモデルで端から端まで確かめる。モデルは呼ばない。"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory, create_text_message_object
from a2a.client.errors import A2AClientHTTPError
from a2a.types import Message, Role, Task, TaskState, TaskStatusUpdateEvent
from google.adk.a2a.converters.part_converter import convert_genai_part_to_a2a_part
from google.adk.apps import App, ResumabilityConfig
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from samples.a2a.test_a2a import free_port
from samples.a2a.test_a2a_client import pending_call
from samples.expense_hub import store
from samples.expense_hub.approval_server import agent as approval
from samples.expense_hub.approval_server.tools import approval_route, request_approval
from samples.expense_hub.expense_server import agent as expense
from samples.expense_hub.expense_server.tools import register_expense
from samples.expense_hub.orchestrator.agent import build_orchestrator

EXPENSE_URL = "http://expense.local"
APPROVAL_URL = "http://approval.local"
PROJECT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPENSE_HUB_DB", str(tmp_path / "hub.sqlite3"))


def reply(part: types.Part) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[part]))


def call(name: str, **args) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def last_parts(req: LlmRequest) -> tuple[types.FunctionResponse | None, str]:
    parts = req.contents[-1].parts or []
    response = next((p.function_response for p in parts if p.function_response), None)
    return response, "".join(p.text or "" for p in parts)


class ExpenseScript(BaseLlm):
    """「〇〇円」を交通費として登録し、経費 ID と金額を返す。"""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        response, text = last_parts(llm_request)
        if response is not None:
            r = response.response
            yield reply(
                types.Part(
                    text=f"登録しました。経費ID {r['expense_id']}、金額 {r['amount']}円"
                )
            )
            return
        amount = int(re.search(r"(\d+)円", text).group(1))
        yield reply(
            call(
                "register_expense",
                expense_date="2026-09-01",
                category="交通費",
                amount=amount,
                description="大阪出張",
            )
        )


class ApprovalScript(BaseLlm):
    """経費 ID で承認を申請し、承認待ちなら入力待ちで尋ね、返答を判断として記録する。"""

    received: list[str] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        response, text = last_parts(llm_request)
        if response is None:
            self.received = [*self.received, text]
            found = re.search(r"EXP-\d{4}", text)
            if found is None:
                yield reply(types.Part(text="経費IDを教えてください"))
                return
            yield reply(
                call("request_approval", expense_id=found.group(0), reason="大阪出張")
            )
            return
        r = response.response
        if response.name == "request_approval" and r["status"] == "pending":
            message = (
                f"{r['approver']}の承認が必要です（{r['amount']}円）。承認しますか"
            )
            yield reply(call("adk_request_input", message=message))
        elif response.name == "request_approval":
            yield reply(types.Part(text=f"自動承認しました（{r['amount']}円）"))
        elif response.name == "adk_request_input":
            expense_id = next(
                p.function_call.args["expense_id"]
                for c in llm_request.contents
                for p in c.parts or []
                if p.function_call and p.function_call.name == "request_approval"
            )
            yield reply(
                call(
                    "record_decision",
                    expense_id=expense_id,
                    approved="承認" in r["answer"],
                )
            )
        else:
            yield reply(types.Part(text=f"判断を記録しました: {r['status']}"))


class HubScript(BaseLlm):
    """登録を expense_agent に頼み、結果を受けたら承認に回す。"""

    approval_as_tool: bool = False
    tool_results: list[dict] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        response, text = last_parts(llm_request)
        if response is None:
            yield reply(call("expense_agent", request=text))
            return
        self.tool_results = [*self.tool_results, {response.name: response.response}]
        if response.name == "expense_agent" and self.approval_as_tool:
            request = (
                f"{response.response['result']}。この経費を大阪出張の理由で承認申請して"
            )
            yield reply(call("approval_agent", request=request))
        elif response.name == "expense_agent":
            yield reply(call("transfer_to_agent", agent_name="approval_agent"))
        else:
            yield reply(types.Part(text="終わりました"))


@pytest.fixture
async def hub() -> AsyncGenerator[tuple[httpx.AsyncClient, ApprovalScript], None]:
    approval_model = ApprovalScript(model="gemini-3.8-flash")
    expense_app = expense.create_app(
        expense.build_agent(ExpenseScript(model="gemini-3.8-flash")), EXPENSE_URL
    )
    approval_app = approval.create_app(
        approval.build_agent(approval_model), APPROVAL_URL
    )
    async with AsyncExitStack() as stack:
        for app in (expense_app, approval_app):
            await stack.enter_async_context(app.router.lifespan_context(app))
        hc = await stack.enter_async_context(
            httpx.AsyncClient(
                mounts={
                    EXPENSE_URL: httpx.ASGITransport(app=expense_app),
                    APPROVAL_URL: httpx.ASGITransport(app=approval_app),
                }
            )
        )
        yield hc, approval_model


async def start(
    hc: httpx.AsyncClient,
    approval_as_tool: bool,
    streaming: bool = True,
    resumable: bool = True,
) -> tuple[InMemoryRunner, HubScript]:
    model = HubScript(model="gemini-3.8-flash", approval_as_tool=approval_as_tool)
    orchestrator = build_orchestrator(
        model=model,
        expense_card=f"{EXPENSE_URL}/.well-known/agent-card.json",
        approval_card=f"{APPROVAL_URL}/.well-known/agent-card.json",
        a2a_client_factory=ClientFactory(
            ClientConfig(httpx_client=hc, streaming=streaming)
        ),
        approval_as_tool=approval_as_tool,
    )
    # 再開を有効にしないと、利用者の答えは問いを出したサブエージェントではなくルートに回る。
    config = ResumabilityConfig(is_resumable=True) if resumable else None
    runner = InMemoryRunner(
        app=App(name="hub", root_agent=orchestrator, resumability_config=config)
    )
    await runner.session_service.create_session(
        app_name="hub", user_id="u", session_id="s"
    )
    return runner, model


async def turn_events(
    runner: InMemoryRunner, part: types.Part
) -> list[tuple[str, types.Part]]:
    out: list[tuple[str, types.Part]] = []
    message = types.Content(role="user", parts=[part])
    async for ev in runner.run_async(user_id="u", session_id="s", new_message=message):
        out += [(ev.author, p) for p in (ev.content.parts if ev.content else [])]
    return out


async def turn(runner: InMemoryRunner, part: types.Part) -> list[types.Part]:
    return [p for _, p in await turn_events(runner, part)]


def questions(parts: list[types.Part]) -> list[types.FunctionCall]:
    return [
        p.function_call
        for p in parts
        if p.function_call and p.function_call.name == "adk_request_input"
    ]


@pytest.mark.parametrize(
    ("amount", "want"),
    [(5_000, "auto"), (5_001, "manager"), (50_000, "manager"), (50_001, "director")],
    ids=["5000円は自動", "5001円から上長", "50000円まで上長", "50001円から部長"],
)
def test_approval_route_boundaries(amount: int, want: str) -> None:
    assert approval_route(amount) == want


def test_approval_uses_stored_amount() -> None:
    """承認の Spoke は申請の文面ではなく、保存した金額で経路を決める。"""
    registered = register_expense("2026-09-01", "交通費", 40_000, "大阪出張")
    result = request_approval(registered["expense_id"], "大阪出張")
    assert (result["status"], result["route"], result["amount"]) == (
        "pending",
        "manager",
        40_000,
    )


async def test_small_expense_is_auto_approved(hub) -> None:
    hc, approval_model = hub
    runner, _ = await start(hc, approval_as_tool=False)
    parts = await turn(runner, types.Part(text="交通費3000円を登録して承認申請して"))
    assert "自動承認しました（3000円）" in [p.text for p in parts if p.text]
    # 承認の Spoke には、前の段の経費 ID が会話の文として届く。
    assert "EXP-0001" in approval_model.received[-1]
    assert store.get("EXP-0001").approval == "approved"


@pytest.mark.parametrize("resumable", [True, False], ids=["再開あり", "再開なし"])
async def test_sub_agent_surfaces_question_but_drops_answer(
    hub, resumable: bool
) -> None:
    """承認をサブエージェントにすると、問いは利用者まで届くが、答えは Spoke に届かず承認待ちのまま残る。"""
    hc, approval_model = hub
    runner, model = await start(hc, approval_as_tool=False, resumable=resumable)
    parts = await turn(runner, types.Part(text="交通費40000円を登録して承認申請して"))
    # ストリーミングでは、同じ問いが同じ ID で 2 回届く。
    first, second = questions(parts)
    assert first.id == second.id
    assert "上長の承認が必要です（40000円）" in first.args["message"]

    answer = types.FunctionResponse(
        id=first.id, name=first.name, response={"answer": "承認します"}
    )
    # ADK 2.2.0 は承認のノードを完了済みとして飛ばし、イベントを 1 件も返さない。
    assert await turn(runner, types.Part(function_response=answer)) == []
    assert len(approval_model.received) == 1
    assert len(model.tool_results) == 1
    assert store.get("EXP-0001").approval == "pending"


async def test_direct_client_completes_input_required_by_requester(hub) -> None:
    """承認の Spoke を直接呼ぶと教材の状態遷移になる。答えるのは呼び出し元で、申請者でも承認できる。"""
    hc, _ = hub
    expense_id = register_expense("2026-09-01", "交通費", 40_000, "大阪出張")[
        "expense_id"
    ]
    client = await ClientFactory.connect(
        APPROVAL_URL,
        client_config=ClientConfig(httpx_client=hc, streaming=True),
    )
    states: list[str] = []

    async def run(message: Message) -> Task:
        last = None
        async for task, update in client.send_message(message):
            last = task
            if update is None or isinstance(update, TaskStatusUpdateEvent):
                state = task.status.state.value
                if not states or states[-1] != state:
                    states.append(state)
        return last

    task = await run(create_text_message_object(content=f"{expense_id} を承認申請して"))
    assert task.status.state == TaskState.input_required
    call = pending_call(task)
    genai_part = types.Part(
        function_response=types.FunctionResponse(
            id=call["id"], name=call["name"], response={"answer": "承認します"}
        )
    )
    answer = Message(
        role=Role.user,
        parts=[convert_genai_part_to_a2a_part(genai_part)],
        message_id="answer-1",
        task_id=task.id,
        context_id=task.context_id,
    )
    task = await run(answer)
    assert task.status.state == TaskState.completed
    assert states == ["submitted", "working", "input-required", "working", "completed"]
    assert store.get(expense_id).approval == "approved"


@pytest.mark.parametrize(
    ("streaming", "echoed", "asked"),
    [(True, True, 2), (False, False, 1)],
    ids=[
        "ストリーミングでは受けた会話が折り返され問いが2回",
        "ストリーミングなしでは折り返しなく問いが1回",
    ],
)
async def test_streaming_echoes_the_forwarded_conversation(
    hub, streaming: bool, echoed: bool, asked: int
) -> None:
    """ストリーミングで呼ぶと、Spoke に渡した会話（For context の行を含む）が Spoke の出力として戻る。"""
    hc, _ = hub
    runner, _ = await start(hc, approval_as_tool=False, streaming=streaming)
    events = await turn_events(
        runner, types.Part(text="交通費40000円を登録して承認申請して")
    )
    approval_texts = [
        p.text for author, p in events if author == "approval_agent" and p.text
    ]
    assert ("交通費40000円を登録して承認申請して" in approval_texts) is echoed
    assert any("called tool `expense_agent`" in t for t in approval_texts) is echoed
    assert len(questions([p for _, p in events])) == asked


async def test_agent_tool_loses_input_required(hub) -> None:
    """教材の構成（承認も AgentTool）では、入力待ちの問いが空の結果になり、承認待ちのまま残る。"""
    hc, _ = hub
    runner, model = await start(hc, approval_as_tool=True)
    parts = await turn(runner, types.Part(text="交通費40000円を登録して承認申請して"))
    assert not questions(parts)
    assert {"approval_agent": {"result": ""}} in model.tool_results
    assert store.get("EXP-0001").approval == "pending"


async def test_unreachable_spoke_is_a2a_http_error() -> None:
    """Spoke が起動していないとき、届くのは ConnectionRefusedError ではなく A2AClientHTTPError の 503。"""
    with pytest.raises(A2AClientHTTPError) as caught:
        await ClientFactory.connect(f"http://127.0.0.1:{free_port()}")
    assert caught.value.status_code == 503


def test_run_servers_starts_both_and_stops_on_sigint(tmp_path: Path) -> None:
    """起動スクリプトは 2 つの Spoke を立て、SIGINT ですべて止める。"""
    ports = {"EXPENSE_PORT": free_port(), "APPROVAL_PORT": free_port()}
    env = (
        os.environ
        | {k: str(v) for k, v in ports.items()}
        | {"EXPENSE_HUB_DB": str(tmp_path / "db")}
    )
    env.pop("GEMINI_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "samples.expense_hub.run_servers"],
        cwd=PROJECT,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        names = {}
        for port in ports.values():
            for _ in range(100):
                try:
                    card = httpx.get(
                        f"http://127.0.0.1:{port}/.well-known/agent-card.json",
                        timeout=0.2,
                    )
                    names[port] = card.json()["name"]
                    break
                except httpx.TransportError:
                    time.sleep(0.1)
        assert sorted(names.values()) == ["approval_agent", "expense_agent"]
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0
    assert out.count("を停止しました") == 2
    for port in ports.values():
        with pytest.raises(httpx.TransportError):
            httpx.get(
                f"http://127.0.0.1:{port}/.well-known/agent-card.json", timeout=0.5
            )
