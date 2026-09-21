"""Hub-Spoke のオーケストレーター。adk web ./orchestrator で起動する。

経費精算は AgentTool で呼び、承認はサブエージェントにする。
AgentTool で包むと、承認の入力待ちの問いが空の結果になる。サブエージェントなら問いは利用者に届く。
ただし ADK 2.2.0 では、その答えが承認の Spoke に届かない。入力待ちを完了させるには Spoke を直接呼ぶ。
"""

from __future__ import annotations

import os

from google.adk import Agent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.models import BaseLlm
from google.adk.tools.agent_tool import AgentTool

MODEL = "gemini-3.8-flash"
CARD_PATH = "/.well-known/agent-card.json"

INSTRUCTION = """あなたは経費精算の窓口です
- 経費の登録と照会は expense_agent ツールに頼む
- 承認の申請は approval_agent に引き継ぐ。引き継ぐ前に、登録した経費 ID を会話に書いておく
- 登録と承認の両方を頼まれたら、先に登録し、返ってきた経費 ID で承認を申請する"""


def build_orchestrator(
    model: str | BaseLlm = MODEL,
    expense_card: object | None = None,
    approval_card: object | None = None,
    a2a_client_factory: object | None = None,
    approval_as_tool: bool = False,
) -> Agent:
    """approval_as_tool は教材の構成（承認も AgentTool）を再現するためのもの。"""
    expense_card = expense_card or (
        os.environ.get("EXPENSE_AGENT_URL", "http://localhost:8001") + CARD_PATH
    )
    approval_card = approval_card or (
        os.environ.get("APPROVAL_AGENT_URL", "http://localhost:8002") + CARD_PATH
    )
    expense = RemoteA2aAgent(
        name="expense_agent",
        description="経費データの登録と照会",
        agent_card=expense_card,
        a2a_client_factory=a2a_client_factory,
    )
    approval = RemoteA2aAgent(
        name="approval_agent",
        description="経費の承認の申請",
        agent_card=approval_card,
        a2a_client_factory=a2a_client_factory,
    )
    tools = [AgentTool(expense)]
    sub_agents = []
    if approval_as_tool:
        tools.append(AgentTool(approval))
    else:
        sub_agents.append(approval)
    return Agent(
        name="orchestrator",
        model=model,
        instruction=INSTRUCTION,
        tools=tools,
        sub_agents=sub_agents,
    )


root_agent = build_orchestrator()
