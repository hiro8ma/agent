"""静的、動的、グローバルの Instruction を一つの App で使い分ける。"""

from __future__ import annotations

from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.apps import App
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin

MODEL = "gemini-3.8-flash"

GLOBAL_POLICY = (
    "顧客の認証情報や決済情報を要求しません。"
    "確証のない内容を事実として断定しません。"
    "担当外の操作は実行せず、人間の担当者へ引き継ぎます。"
)


def support_instruction(ctx: ReadonlyContext) -> str:
    """利用者と案件の State から、問い合わせ固有の指示を作る。"""
    display_name = ctx.state.get("user:display_name", "お客様")
    issue_kind = ctx.state.get("issue_kind", "未分類")
    return (
        "あなたは技術サポート担当です。"
        f"対応中の利用者は {display_name} です。"
        f"問い合わせ分類は {issue_kind} です。"
        "利用者が提示した事実とツール結果だけを使って回答してください。"
    )


order_agent = Agent(
    name="order_agent",
    model=MODEL,
    description="注文状況と返品手続きを扱う",
    instruction=(
        "注文番号を確認し、注文ツールの結果だけを使って案内してください。"
        "注文内容を推測しないでください。"
    ),
)

support_agent = Agent(
    name="support_agent",
    model=MODEL,
    description="技術的な問い合わせを扱う",
    instruction=support_instruction,
)

root_agent = Agent(
    name="support_router",
    model=MODEL,
    description="問い合わせを注文管理または技術サポートへ振り分ける",
    instruction="問い合わせ内容に対応する担当へ振り分けてください。",
    sub_agents=[order_agent, support_agent],
)

app = App(
    name="support_context_demo",
    root_agent=root_agent,
    plugins=[GlobalInstructionPlugin(global_instruction=GLOBAL_POLICY)],
)
