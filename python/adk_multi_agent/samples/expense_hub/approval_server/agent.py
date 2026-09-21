"""承認の Spoke。上長と部長の承認は request_input で入力待ちにする。

adk web ではこのファイルの root_agent を、A2A では create_app を使う。
"""

from __future__ import annotations

import os

from google.adk import Agent
from google.adk.models import BaseLlm
from google.adk.tools import request_input

from samples.expense_hub.approval_server.tools import record_decision, request_approval

MODEL = "gemini-3.8-flash"
PORT = 8002

INSTRUCTION = """あなたは経費の承認を管理する担当です
- 承認の申請には request_approval を使う。金額はツールが経費 ID から引くので、依頼の文面の金額は使わない
- 結果が approved なら、自動承認したと伝える
- 結果が pending なら、adk_request_input で承認者に承認するかを尋ね、返答を record_decision で記録する
- 経費 ID か申請の理由が無ければ聞き返す"""


def build_agent(model: str | BaseLlm = MODEL) -> Agent:
    return Agent(
        name="approval_agent",
        model=model,
        description="経費の承認の申請を受け付け、金額に応じて自動承認か承認者への確認を行う",
        instruction=INSTRUCTION,
        tools=[request_approval, record_decision, request_input],
    )


root_agent = build_agent()


# ここから下だけが A2A に依存する。


def agent_card(base_url: str):
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    return AgentCard(
        name="approval_agent",
        description="経費の承認の申請を受け付け、金額に応じて自動承認か承認者への確認を行う",
        url=base_url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="request_approval",
                name="承認の申請",
                description="経費 ID と申請の理由を受け取る。5,000 円以下は自動承認、50,000 円以下は上長、それを超えると部長の確認を入力待ちで求める",
                tags=["expense", "approval"],
                examples=["EXP-0001 を大阪出張の交通費として承認申請して"],
            ),
        ],
    )


def create_app(agent: Agent | None = None, base_url: str | None = None):
    from google.adk.a2a.utils.agent_to_a2a import to_a2a

    base_url = base_url or os.environ.get(
        "APPROVAL_AGENT_URL", f"http://localhost:{PORT}"
    )
    return to_a2a(agent or root_agent, agent_card=agent_card(base_url))
