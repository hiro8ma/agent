"""実行結果のレビュー。

判定を文字列で受け取ると、約束と解釈が二重に必要になる。
プロンプトで「`VERDICT: PASS` と書け」と指示し、
受け取る側でその形を解析する。どちらかがずれると壊れる。

実際にずれた。指示は `VERDICT: PASS` なのに、判定側は
`"ok"` で始まるかを見ていた。自己修正が常に False になり、
コードは動き、例外も出ず、何も直らないまま試行回数だけ増えた。

**bool で受け取れば、その経路自体が無くなる。**
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

    判定が読み取れない場合は False にする。
    読めなかったものを True に倒すと、
    壊れた結果がそのまま最終レポートへ流れる。
    やり直しの費用より、誤った結論のほうが高い。
    """

    completed = bool(re.search(r"COMPLETED\s*:\s*(true|yes|pass)", text, re.I))
    m = re.search(r"OBSERVATION\s*:\s*(.+)", text, re.S | re.I)
    observation = m.group(1).strip() if m else text.strip()
    return Review(observation=observation[:1000], is_completed=completed)
