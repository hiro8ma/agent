"""スキルの共有と再利用。

共通ライブラリとドメイン別のスキルを別ディレクトリに置き、必要な組み合わせだけを読む。

    library/common/    どのエージェントでも使う手順
    library/domain/    業務ごとの手順。版違いはここに並べる

`list_skills_in_dir` は直下しか見ない。
`common/auth/` のように入れ子にすると「SKILL.md が無い」と警告してスキップされる。
そのため階層で分けるのではなく、ディレクトリを分けて読み込み側で合成する。

版は名前に入れる。教材にある `order-management/v1/` の形は、
name とディレクトリ名の一致という制約に触れて読み込み時に落ちる。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from google.adk.skills import Skill, load_skill_from_dir

# 役割ごとに使えるスキル。名前は版を含めない論理名にする。
ROLE_SKILLS: dict[str, tuple[str, ...]] = {
    "free": ("notify-user",),
    "standard": ("notify-user", "order-management"),
    "premium": ("notify-user", "order-management"),
    "admin": ("notify-user", "order-management", "admin-operations"),
}

# 本文が参照するツール名。`analyze_reviews` のようにバッククォートで囲んだ英小文字の語を拾う。
_TOOL_REF = re.compile(r"`([a-z][a-z0-9_]{2,})`")


def load_library(*dirs: Path) -> dict[str, Skill]:
    """複数のディレクトリからスキルを読み、名前で引ける形にまとめる。

    同じ名前が 2 つあれば、どちらが使われるか読み手に分からなくなるので落とす。
    """
    library: dict[str, Skill] = {}
    for base in dirs:
        for entry in sorted(Path(base).iterdir()):
            if not (entry / "SKILL.md").is_file():
                continue
            skill = load_skill_from_dir(entry)
            name = skill.frontmatter.name
            if name in library:
                raise ValueError(f"スキル名が重複している: {name}")
            library[name] = skill
    return library


def select_version(user_id: str, versions: Sequence[str] = ("v1", "v2")) -> str:
    """利用者ごとに版を決める。同じ利用者には常に同じ版を返す。

    乱数で振ると、同じ利用者が呼ぶたびに手順が変わる。
    """
    if not versions:
        raise ValueError("versions が空")
    digest = hashlib.sha256(user_id.encode("utf-8")).digest()
    return versions[digest[0] % len(versions)]


def resolve_names(
    logical_names: Iterable[str], library: dict[str, Skill], version: str
) -> list[str]:
    """論理名を、ライブラリにある実際のスキル名へ解決する。

    版付きの名前（order-management-v2）があればそれを、無ければ論理名をそのまま返す。
    """
    resolved = []
    for name in logical_names:
        versioned = f"{name}-{version}"
        resolved.append(versioned if versioned in library else name)
    return resolved


def skills_for_role(
    role: str, library: dict[str, Skill], version: str = "v1"
) -> tuple[list[Skill], list[str]]:
    """役割に応じて使えるスキルと、ライブラリに無い名前を返す。

    未知の役割は最小の集合として扱う。権限の判断で迷ったら狭い方へ倒す。
    """
    logical = ROLE_SKILLS.get(role, ROLE_SKILLS["free"])
    names = resolve_names(logical, library, version)
    found = [library[n] for n in names if n in library]
    missing = [n for n in names if n not in library]
    return found, missing


def referenced_tools(skill: Skill) -> set[str]:
    """SKILL.md の本文が名前を挙げているツールを拾う。"""
    return set(_TOOL_REF.findall(skill.instructions))


def missing_tools(skill: Skill, available: Iterable[str]) -> set[str]:
    """本文が挙げているのに、エージェントへ渡していないツールを返す。

    教材は「整合性は機械的に検出できない」と書くが、本文の参照は拾えるので検出できる。
    拾えるのは書き方の規約（バッククォート）に沿った部分だけなので、完全ではない。
    """
    return referenced_tools(skill) - set(available)
