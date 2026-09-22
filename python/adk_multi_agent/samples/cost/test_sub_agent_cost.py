"""sub_agents に分けたときの、1 つの問い合わせあたりの LLM 呼び出しと入力の大きさを確かめる。モデルは呼ばない。"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk import Agent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

QUESTION = "パスワードの再設定方法を教えてください"


class Calls:
    """全エージェントの呼び出しを 1 か所に記録する。"""

    def __init__(self) -> None:
        self.log: list[tuple[str, int, int]] = []


def size(req: LlmRequest) -> tuple[int, int]:
    instruction = str(req.config.system_instruction or "")
    contents = sum(
        len(p.text or "") + len(str(p.function_call or p.function_response or ""))
        for c in req.contents
        for p in c.parts or []
    )
    return len(instruction), contents


class Script(BaseLlm):
    """transfer_to があれば委譲し、無ければ答える。"""

    transfer_to: str | None = None
    calls: object = None

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        self.calls.log.append((self.model, *size(llm_request)))
        if self.transfer_to:
            part = types.Part(
                function_call=types.FunctionCall(
                    name="transfer_to_agent", args={"agent_name": self.transfer_to}
                )
            )
        else:
            part = types.Part(text="設定画面から再設定できます")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


async def ask(agent: Agent) -> None:
    runner = InMemoryRunner(agent=agent, app_name="support")
    await runner.session_service.create_session(
        app_name="support", user_id="u", session_id="s"
    )
    message = types.Content(role="user", parts=[types.Part(text=QUESTION)])
    async for _ in runner.run_async(user_id="u", session_id="s", new_message=message):
        pass


ROOT = "カスタマーサポートシステムのルートエージェントです。ユーザーの問い合わせ内容に応じて、対応するサブエージェントに処理を委譲してください。"
ROUTER = "ユーザーの問い合わせを分類し、対応するサブエージェントに振り分けます。分類カテゴリ: 一般質問/技術質問/アカウント問題/クレーム"
RESEARCH = "複雑な調査タスクを担当します。十分に検討した上で回答してください。"
FAQ = "よくある質問に定型の回答を返します。"


def book_tree(calls: Calls, root_to: str, router_to: str | None) -> Agent:
    """教材の構成。ルートの下にルーター、調査、FAQ が並ぶ。"""

    def sub(name: str, instruction: str, to: str | None = None) -> Agent:
        return Agent(
            name=name,
            model=Script(model=name, transfer_to=to, calls=calls),
            instruction=instruction,
            description=instruction[:30],
        )

    return Agent(
        name="support_system",
        model=Script(model="support_system", transfer_to=root_to, calls=calls),
        instruction=ROOT,
        sub_agents=[
            sub("router_agent", ROUTER, router_to),
            sub("research_agent", RESEARCH),
            sub("faq_agent", FAQ),
        ],
    )


async def test_single_agent_answers_in_one_call() -> None:
    calls = Calls()
    await ask(
        Agent(
            name="support",
            model=Script(model="support", calls=calls),
            instruction=ROOT + FAQ,
        )
    )
    assert [m for m, _, _ in calls.log] == ["support"]


async def test_book_tree_adds_calls_and_repeats_history() -> None:
    """教材の構成では、ルートの判断の分だけ呼び出しが増え、委譲先にも会話の全体が渡る。"""
    direct = Calls()
    await ask(book_tree(direct, root_to="faq_agent", router_to=None))
    via_router = Calls()
    await ask(book_tree(via_router, root_to="router_agent", router_to="faq_agent"))

    assert [m for m, _, _ in direct.log] == ["support_system", "faq_agent"]
    assert [m for m, _, _ in via_router.log] == [
        "support_system",
        "router_agent",
        "faq_agent",
    ]

    # 委譲先の入力には、利用者の問いと、前のエージェントの委譲のやり取りが含まれる。
    (_, _, root_contents), (_, _, faq_contents) = direct.log
    assert faq_contents > root_contents

    # Instruction には、委譲できる相手の一覧が足される。ルートの Instruction は教材の文より長くなる。
    root_instruction = direct.log[0][1]
    assert root_instruction > len(ROOT)
