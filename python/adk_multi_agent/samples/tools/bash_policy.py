"""ExecuteBashTool の許可規則を狭める。

既定の BashToolPolicy は全コマンドを許可し、演算子を 1 つも塞がない。
前方一致の許可リストだけでは ; や $() で連結して迂回される。
ここで作る規則も境界ではない。本当の境界はサンドボックス側に置く。
"""

from __future__ import annotations

from google.adk.tools.bash_tool import BashToolPolicy

# 連結とリダイレクトと置換。1 つでも残すと許可リストを迂回される。
BLOCKED_OPERATORS = (";", "&&", "||", "|", "$(", "`", ">", "<", "\n", "&")


def strict_policy(*commands: str) -> BashToolPolicy:
    """コマンド名を空白つきで許可し、演算子を塞ぐ。

    前方一致は文字列の比較なので "ls" だと lsblk も通る。
    "ls " にすると引数なしの ls は通らなくなる。この検証器では
    「そのコマンド名ちょうど」を表せない。
    """
    return BashToolPolicy(
        allowed_command_prefixes=tuple(f"{c} " for c in commands),
        blocked_operators=BLOCKED_OPERATORS,
        timeout_seconds=10,
        max_memory_bytes=256 * 1024 * 1024,
        max_file_size_bytes=10 * 1024 * 1024,
        max_child_processes=16,
    )
