"""分析計画。

サブタスクではなく仮説を立てさせる。「〜を出す」は作業で、
出したあとに何が言えるかが決まらない。
グラフ種別も先に決め、生成の自由度を下げる。

達成条件は 2 階層。Plan.achievement がレポートの、
Program.achievement_condition がレビューの基準になる。
片方だけだと、個々のコードは通るのに要求に答えていない結果が出る。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class Task(BaseModel):
    """検証する仮説 1 件。"""

    hypothesis: str = Field(default="", description="検証可能な仮説")
    purpose: str = Field(default="", description="仮説の検証目的")
    description: str = Field(default="", description="分析方針と可視化対象")
    chart_type: str = Field(default="", description="グラフ想定")


class Plan(BaseModel):
    """タスク要求に対する分析計画。"""

    purpose: str = Field(default="", description="問い合わせ目的")
    achievement: str = Field(default="", description="推測される達成条件")
    tasks: list[Task] = Field(default_factory=list)


_FIELDS = {
    "仮説": "hypothesis",
    "目的": "purpose",
    "方針": "description",
    "グラフ": "chart_type",
}


def parse_plan(text: str) -> Plan:
    """計画文を Plan に分解する。

    仮説が 1 件も読み取れない場合、呼び出し側が
    元の要求そのものを 1 件の仮説として扱う。
    計画が空でも分析自体は進められるようにする。
    """

    def head(name: str) -> str:
        m = re.search(rf"^\s*{name}\s*[:：]\s*(.+)$", text, re.M)
        return m.group(1).strip() if m else ""

    tasks: list[Task] = []
    # 「仮説」で始まる行を区切りにして、続く行から属性を拾う。
    for block in re.split(r"\n(?=\s*(?:[-*]\s*)?仮説\s*[:：])", text):
        if not re.search(r"仮説\s*[:：]", block):
            continue
        values = {}
        for label, key in _FIELDS.items():
            m = re.search(rf"{label}\s*[:：]\s*(.+)", block)
            if m:
                values[key] = m.group(1).strip()
        if values.get("hypothesis"):
            tasks.append(Task(**values))

    return Plan(purpose=head("目的"), achievement=head("達成条件"), tasks=tasks)
