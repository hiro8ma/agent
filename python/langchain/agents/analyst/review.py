"""実行結果のレビュー。

判定を bool で受ける。文字列だと、プロンプトの約束と
受け取る側の解析がずれても例外にならない。

本番では Structured Outputs が形を保証する。
ここでは行の解析に落とすが、フィールド名は同じにしておく。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class Review(BaseModel):
    """1 回の実行結果に対する判定。"""

    observation: str = Field(default="", description="コードに対するフィードバック")
    is_completed: bool = Field(default=False, description="実行結果がタスク要求を満たすか")


def parse_review(text: str) -> Review:
    """レビュー文を Review に分解する。

    読み取れない場合は False に倒す。True にすると
    壊れた結果がそのまま最終レポートへ流れる。
    """

    completed = bool(re.search(r"COMPLETED\s*:\s*(true|yes|pass)", text, re.I))
    m = re.search(r"OBSERVATION\s*:\s*(.+)", text, re.S | re.I)
    observation = m.group(1).strip() if m else text.strip()
    return Review(observation=observation[:1000], is_completed=completed)
