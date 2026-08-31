"""コード生成の中間生成物。

達成条件を生成時に宣言させ、レビューの判定基準にする。
コードだけを返させると、何をもって正しいとするかが決まらず、
例外が出なければ通す判定に落ちる。

本番では Structured Outputs が形を保証する。
ここでは見出しで区切った形式を解析し、欠けた項目は空文字にする。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class Program(BaseModel):
    """1 回のコード生成の結果。"""

    achievement_condition: str = Field(default="", description="要求の達成条件")
    execution_plan: str = Field(default="", description="実行計画")
    code: str = Field(default="", description="生成対象となるコード")


def parse_program(text: str) -> Program:
    """生成された文章を Program に分解する。

    見出しが欠けていても落ちない。
    コードだけは必ず取り出す。無いと実行そのものができない。
    """

    def section(name: str) -> str:
        m = re.search(rf"##\s*{name}\s*\n(.*?)(?=\n##\s|\Z)", text, re.S)
        return m.group(1).strip() if m else ""

    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = blocks[0].strip() if blocks else ""
    if not code:
        # 見出しもコードブロックも無ければ、全文をコードとみなす。
        # 説明文が混ざったまま実行されて SyntaxError になるが、
        # 黙って空を返すより、失敗として次の試行へ渡すほうがよい。
        code = text.strip()

    return Program(
        achievement_condition=section("達成条件"),
        execution_plan=section("実行計画"),
        code=code,
    )
