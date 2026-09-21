"""Session / State / Compaction / Memory / RAG を 1 つにまとめたサポート担当。

Compaction は App に設定する。Runner に agent だけを渡すと、Runner が設定の空の App で包み直すので効かない。
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.apps.app import App
from google.adk.models import BaseLlm
from google.adk.runners import Runner
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.retrieval import VertexAiRagRetrieval

from samples.compaction.config import compaction_config
from samples.state.keys import get_user_tier, load_user_state
from samples.support.session_config import (
    create_memory_service,
    create_session_service,
    memory_enabled,
)
from samples.support.tools import get_order_status, search_products

MODEL = "gemini-3.8-flash"
APP_NAME = "customer_support"

TIER_POLICY = {
    "free": "回答は FAQ の範囲に留め、個別の対応が要る場合は問い合わせ窓口を案内する",
    "standard": "注文の状況を調べて答える。返品の手続きは窓口を案内する",
    "premium": "注文の状況を調べて答え、返品の手続きもこの場で案内する",
}

# 情報源の優先順位と、食い違ったときの扱い。検索の順番はツールのコードが決めるので、ここでは書かない。
SOURCE_POLICY = """## 情報源の扱い
- 商品の仕様、返品や保証の規定は search_products で調べる
- 注文の状況は get_order_status で調べる
- 過去の会話（PAST_CONVERSATIONS）は、この利用者の好みや事情を知るためだけに使う
- 規定について過去の会話と検索結果が食い違ったら、検索結果を採用し、以前と変わったことを利用者に伝える
- 好みや事情について食い違ったら、過去の会話を採用する"""


def build_instruction(ctx: ReadonlyContext) -> str:
    """会員区分に応じて答え方を変える。

    区分は user: の State から読むが、書き込むツールは持たせない。
    モデルが区分を書き換えられると、次の Session から上位の対応を受けられてしまう。
    """
    user = load_user_state(ctx.state)
    tier = get_user_tier(ctx.state)
    return "\n\n".join(
        [
            f"あなたはカスタマーサポート担当です。利用者は {user.name} さん（{tier}）。",
            f"## 対応の範囲\n{TIER_POLICY[tier]}",
            SOURCE_POLICY,
        ]
    )


async def save_session_to_memory(callback_context: CallbackContext) -> None:
    await callback_context.add_session_to_memory()


def build_tools(env: Mapping[str, str]) -> list:
    tools: list = [search_products, get_order_status]
    if memory_enabled(env):
        tools.append(PreloadMemoryTool())
    if corpus := env.get("RAG_CORPUS_ID"):
        tools.append(
            VertexAiRagRetrieval(
                name="product_manuals",
                description="製品マニュアルを検索する",
                rag_corpora=[corpus],
                similarity_top_k=5,
                vector_distance_threshold=0.5,
            )
        )
    return tools


def build_agent(
    env: Mapping[str, str] = os.environ, model: BaseLlm | str = MODEL
) -> Agent:
    """Memory を使わない環境では保存のコールバックを付けない。

    付けたままにすると、Memory のサービスが無い Runner では毎ターン ValueError で失敗する。
    """
    return Agent(
        name="support_agent",
        model=model,
        instruction=build_instruction,
        tools=build_tools(env),
        after_agent_callback=save_session_to_memory if memory_enabled(env) else None,
    )


def build_app(env: Mapping[str, str] = os.environ, model: BaseLlm | str = MODEL) -> App:
    return App(
        name=APP_NAME,
        root_agent=build_agent(env, model),
        events_compaction_config=compaction_config("support", summarizer_model=MODEL),
    )


def create_runner(
    env: Mapping[str, str] = os.environ, model: BaseLlm | str = MODEL
) -> Runner:
    return Runner(
        app=build_app(env, model),
        session_service=create_session_service(env),
        memory_service=create_memory_service(env),
        auto_create_session=True,
    )


app = build_app()
root_agent = app.root_agent
