"""ADK のトレースに何が載るかを、手元の OpenTelemetry の出力先で確かめる。モデルは台本で置き換える。"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator

import pytest
from google.adk import Agent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from opentelemetry import _logs, trace
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

PII = "09012345678"
SPANS = InMemorySpanExporter()
LOGS = InMemoryLogExporter()


@pytest.fixture(scope="module", autouse=True)
def providers() -> None:
    # OpenTelemetry の global の provider はプロセスで 1 度しか設定できない。
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(SPANS))
    trace.set_tracer_provider(tp)
    lp = LoggerProvider()
    lp.add_log_record_processor(SimpleLogRecordProcessor(LOGS))
    _logs.set_logger_provider(lp)


@pytest.fixture(autouse=True)
def clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
        "OTEL_SEMCONV_STABILITY_OPT_IN",
    ):
        monkeypatch.delenv(name, raising=False)
    SPANS.clear()
    LOGS.clear()
    yield


def lookup(query: str) -> dict:
    """顧客を検索する。"""
    del query
    return {"hit": 1}


class Script(BaseLlm):
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        last = llm_request.contents[-1].parts[-1]
        if last.function_response:
            part = types.Part(text="回答です")
        else:
            call = types.FunctionCall(name="lookup", args={"query": f"電話 {PII}"})
            part = types.Part(function_call=call)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


async def run() -> None:
    agent = Agent(
        name="root_agent",
        model=Script(model="gemini-3.8-flash"),
        instruction="x",
        tools=[lookup],
    )
    runner = InMemoryRunner(agent=agent, app_name="ops")
    await runner.session_service.create_session(
        app_name="ops", user_id="u", session_id="s"
    )
    message = types.Content(
        role="user", parts=[types.Part(text=f"{PII} の顧客を調べて")]
    )
    async for _ in runner.run_async(user_id="u", session_id="s", new_message=message):
        pass


def leaked_attributes() -> set[str]:
    return {
        key
        for span in SPANS.get_finished_spans()
        for key, value in (span.attributes or {}).items()
        if PII in str(value)
    }


def leaked_logs() -> int:
    return sum(
        PII in str(r.log_record.body) + str(r.log_record.attributes)
        for r in LOGS.get_finished_logs()
    )


async def test_span_tree() -> None:
    """span の名前は教材の root_agent.run / llm.generate_content ではなく、OTel の GenAI の規約に寄せた名前。ツールの span はモデル呼び出しの span の子になる。"""
    await run()
    spans = {s.context.span_id: s for s in SPANS.get_finished_spans()}

    def path(span) -> str:
        names = []
        while span is not None:
            names.append(span.name)
            span = spans.get(span.parent.span_id) if span.parent else None
        return " > ".join(reversed(names))

    paths = {path(s) for s in spans.values()}
    assert (
        "invocation > invoke_agent root_agent > call_llm > generate_content gemini-3.8-flash"
        in paths
    )
    assert (
        "invocation > invoke_agent root_agent > call_llm > generate_content gemini-3.8-flash > execute_tool lookup"
        in paths
    )


async def test_content_is_in_spans_by_default() -> None:
    """何も設定しなくても、プロンプト、応答、ツールの引数が span の属性に入る。"""
    await run()
    assert leaked_attributes() == {
        "gcp.vertex.agent.llm_request",
        "gcp.vertex.agent.llm_response",
        "gcp.vertex.agent.tool_call_args",
    }


async def test_adk_switch_removes_content_from_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false で span の属性から本文が消える。"""
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "false")
    await run()
    assert leaked_attributes() == set()


@pytest.mark.parametrize(
    ("mode", "opt_in", "logged"),
    [
        ("EVENT_ONLY", None, False),
        ("EVENT_ONLY", "gen_ai_latest_experimental", True),
        ("true", None, True),
    ],
    ids=[
        "EVENT_ONLYだけでは記録しない",
        "実験版の規約を有効にするとEVENT_ONLYが効く",
        "trueなら記録する",
    ],
)
async def test_capture_variable_controls_log_events_only(
    monkeypatch: pytest.MonkeyPatch, mode: str, opt_in: str | None, logged: bool
) -> None:
    """OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT はログのイベントだけを切り替え、span の属性は変えない。"""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", mode)
    if opt_in:
        monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", opt_in)
    await run()
    assert (leaked_logs() > 0) is logged
    assert leaked_attributes()
