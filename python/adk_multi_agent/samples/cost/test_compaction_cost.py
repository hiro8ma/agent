"""Compaction で入力がどれだけ減るかを、台本のモデルで文字数を数えて確かめる。モデルは呼ばない。

文字数はトークン数の代わり。比率を見るためのもので、実際のトークン数ではない。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk import Agent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

TURNS = 20
USER = "問" * 300
ANSWER = "答" * 400
SUMMARY = "【要約】" + "要" * 196


def chars(req: LlmRequest) -> int:
    return sum(len(p.text or "") for c in req.contents for p in c.parts or [])


def summaries(req: LlmRequest) -> int:
    return sum(
        (p.text or "").count("【要約】") for c in req.contents for p in c.parts or []
    )


class Main(BaseLlm):
    prompts: list[int] = []
    last_summaries: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.prompts = [*self.prompts, chars(llm_request)]
        self.last_summaries = summaries(llm_request)
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=ANSWER)])
        )


class Summarizer(BaseLlm):
    inputs: list[int] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.inputs = [*self.inputs, chars(llm_request)]
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=SUMMARY)])
        )


async def run(config: EventsCompactionConfig | None) -> tuple[Main, Summarizer]:
    main = Main(model="main")
    summarizer = Summarizer(model="summarizer")
    if config is not None:
        config.summarizer = LlmEventSummarizer(llm=summarizer)
    app = App(
        name="cost",
        root_agent=Agent(name="a", model=main, instruction="x"),
        events_compaction_config=config,
    )
    runner = InMemoryRunner(app=app)
    await runner.session_service.create_session(
        app_name="cost", user_id="u", session_id="s"
    )
    for _ in range(TURNS):
        message = types.Content(role="user", parts=[types.Part(text=USER)])
        async for _ in runner.run_async(
            user_id="u", session_id="s", new_message=message
        ):
            pass
    return main, summarizer


async def test_compaction_reduces_but_summaries_accumulate() -> None:
    """interval=3 / overlap=1 で 20 ターン。最終ターンの入力は減るが、要約は 1 つにまとまらず積み上がる。"""
    plain, _ = await run(None)
    compacted, summarizer = await run(
        EventsCompactionConfig(compaction_interval=3, overlap_size=1)
    )

    assert len(plain.prompts) == len(compacted.prompts) == TURNS
    # 圧縮しなければ、最終ターンの入力は 20 ターン分の原文。
    assert plain.prompts[-1] == TURNS * len(USER) + (TURNS - 1) * len(ANSWER)

    # 要約は範囲ごとに 1 つずつ入り、置き換えられずに並ぶ。
    assert compacted.last_summaries == len(summarizer.inputs) == 6
    last_ratio = compacted.prompts[-1] / plain.prompts[-1]
    main_ratio = sum(compacted.prompts) / sum(plain.prompts)
    with_summarizer = (sum(compacted.prompts) + sum(summarizer.inputs)) / sum(
        plain.prompts
    )
    print(
        f"最終ターン {last_ratio:.2f} / 本体の合計 {main_ratio:.2f} / 要約の入力を含む合計 {with_summarizer:.2f}"
    )
    assert last_ratio < 0.4
    # 要約器への入力（overlap の分を含む）は、本体で減らした分を一部打ち消す。
    assert with_summarizer > main_ratio
