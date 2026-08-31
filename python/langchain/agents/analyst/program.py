"""コード生成の中間生成物。

コードだけを返させると、レビューの基準が曖昧になる。
「タスクに正しく答えられているか」を判定しようとしても、
何をもって正しいとするかが決まっていない。

**達成条件を生成時に宣言させる。**
レビューはそれを基準に判定できるようになる。

仮説を立てるときに検証計画も立てるのと同じ構造になる。
計画が無いと、実行はできるのに何も測れない。

本番では OpenAI の Structured Outputs で形を保証する。
ここでは見出しで区切った形式を解析する。
形式を守らないモデルでも落ちないよう、欠けた項目は空文字にする。
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
