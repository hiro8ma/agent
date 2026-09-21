"""経費精算の Spoke。エージェントの定義と、A2A での公開を分けて置く。

adk web ではこのファイルの root_agent を、A2A では create_app を使う。
"""

from __future__ import annotations

import os

from google.adk import Agent
from google.adk.models import BaseLlm

from samples.expense_hub.expense_server.tools import list_expenses, register_expense

MODEL = "gemini-3.8-flash"
PORT = 8001

INSTRUCTION = """あなたは経費精算の担当です
- 登録の依頼には register_expense を使い、登録した経費 ID と金額を返す
- 照会の依頼には list_expenses を使い、件数と合計を返す
- 日付が無ければ聞き返す。推測で埋めない
- ツールが error を返したら、その内容と直し方を伝える"""


def build_agent(model: str | BaseLlm = MODEL) -> Agent:
    return Agent(
        name="expense_agent",
        model=model,
        description="経費データの登録と、期間を指定した照会を行う",
        instruction=INSTRUCTION,
        tools=[register_expense, list_expenses],
    )


root_agent = build_agent()


# ここから下だけが A2A に依存する。


def agent_card(base_url: str):
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    return AgentCard(
        name="expense_agent",
        description="経費データの登録と、期間を指定した照会を行う",
        url=base_url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="register_expense",
                name="経費の登録",
                description="日付、カテゴリ（交通費 / 宿泊費 / 会議費 / 消耗品費）、金額、用途を受け取り、経費 ID を返す",
                tags=["expense", "register"],
                examples=["2026-09-01 の交通費 12000 円、大阪出張の新幹線を登録して"],
            ),
            AgentSkill(
                id="list_expenses",
                name="経費の照会",
                description="期間を受け取り、その期間の経費の一覧と合計を返す",
                tags=["expense", "query"],
                examples=["2026-09-01 から 2026-09-30 の経費を一覧して"],
            ),
        ],
    )


def create_app(agent: Agent | None = None, base_url: str | None = None):
    from google.adk.a2a.utils.agent_to_a2a import to_a2a

    base_url = base_url or os.environ.get(
        "EXPENSE_AGENT_URL", f"http://localhost:{PORT}"
    )
    return to_a2a(agent or root_agent, agent_card=agent_card(base_url))
