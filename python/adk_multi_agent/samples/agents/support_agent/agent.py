"""コンテキスト最適化したカスタマーサポート。

動的 Instruction、コールバック 4 種、Agent Skills、関数ツールを 1 つに束ねる。

    build_instruction         ティア別の対応方針
    before_model              上限の確認 → 注文情報の注入（この順）
    after_model               禁止語の検査
    before_tool               権限の確認
    after_tool                結果の縮小
    SkillToolset              手順を外部の SKILL.md から読む
    関数ツール 4 つ           実際の操作

スキルはこのパッケージの中に置く。
adk run と adk web はエージェントのディレクトリを独立したパッケージとして読むので、
親や兄弟のディレクトリを参照すると配布時に壊れる。
教材は図では support_agent/skills/、コードでは 1 つ上の skills/ を指していて食い違う。

実行方法:
    adk run samples/agents/support_agent

adk web の情報表示はこの構成では失敗する。
Root は LlmAgent だが、instruction が関数なので AgentInfo の検証で落ちる。
確認は adk run を主経路にする。
"""

from __future__ import annotations

from pathlib import Path

from google.adk import Agent
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset

from .callbacks import (
    authorize_tool_access,
    build_instruction,
    inject_order_context,
    rate_limit_check,
    trim_tool_response,
    validate_response_content,
)
from .tools import cancel_order, get_order_status, get_product_details, search_products

MODEL = "gemini-3.8-flash"
SKILLS_DIR = Path(__file__).resolve().parent / "skills"

order_skill = load_skill_from_dir(SKILLS_DIR / "order-management")
product_skill = load_skill_from_dir(SKILLS_DIR / "product-inquiry")
skill_toolset = SkillToolset(skills=[order_skill, product_skill])

root_agent = Agent(
    name="support_agent",
    model=MODEL,
    description="注文と商品の問い合わせに答えるカスタマーサポート",
    instruction=build_instruction,
    tools=[
        skill_toolset,
        get_order_status,
        cancel_order,
        search_products,
        get_product_details,
    ],
    # リストで渡すと ADK が順に呼び、真を返した時点で止める。合成関数は要らない。
    # 上限で断る要求へ注入しても捨てられるので、上限の確認を先に置く。
    before_model_callback=[rate_limit_check(10), inject_order_context()],
    after_model_callback=[validate_response_content()],
    before_tool_callback=[authorize_tool_access()],
    after_tool_callback=[trim_tool_response()],
)
