"""スキルの共有と再利用の検査。

見るのは 4 つ。

    合成      共通とドメインを別ディレクトリに置き、読み込み側でまとめる
    条件      役割に応じて渡すスキルを変える。未知の役割は最小へ倒す
    版        利用者ごとに決定論的に振り分ける。教材の入れ子の形は落ちる
    整合      SKILL.md が挙げるツール名と、渡すツール名のずれを機械的に拾う
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from google.adk.skills import list_skills_in_dir, load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset

from samples.skills.library import (
    ROLE_SKILLS,
    load_library,
    missing_tools,
    referenced_tools,
    resolve_names,
    select_version,
    skills_for_role,
)

LIBRARY = Path(__file__).parent / "library"
COMMON = LIBRARY / "common"
DOMAIN = LIBRARY / "domain"


def test_library_merges_two_directories():
    library = load_library(COMMON, DOMAIN)
    assert sorted(library) == [
        "notify-user",
        "order-management-v1",
        "order-management-v2",
    ]


def test_duplicate_names_across_directories_are_rejected(tmp_path: Path):
    """同じ名前が 2 か所にあると、どちらが使われるか読み手に分からない。"""
    copy = tmp_path / "copy"
    copy.mkdir()
    shutil.copytree(COMMON / "notify-user", copy / "notify-user")
    with pytest.raises(ValueError, match="重複"):
        load_library(COMMON, copy)


def test_listing_does_not_descend_into_nested_dirs():
    """list_skills_in_dir は直下だけを見る。共通ライブラリを入れ子にできない理由。"""
    assert sorted(list_skills_in_dir(COMMON)) == ["notify-user"]
    assert list_skills_in_dir(LIBRARY) == {}


def test_version_in_subdirectory_fails_to_load(tmp_path: Path):
    """教材の order-management/v1/ の形は読み込み時に落ちる。

    name とディレクトリ名が一致しないため。版は名前へ入れる。
    """
    nested = tmp_path / "order-management" / "v1"
    nested.parent.mkdir()
    shutil.copytree(DOMAIN / "order-management-v1", nested)
    md = nested / "SKILL.md"
    md.write_text(
        md.read_text(encoding="utf-8").replace(
            "name: order-management-v1", "name: order-management"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match directory"):
        load_skill_from_dir(nested)


def test_role_decides_which_skills_are_handed_over():
    library = load_library(COMMON, DOMAIN)

    free, missing = skills_for_role("free", library)
    assert [s.frontmatter.name for s in free] == ["notify-user"]
    assert missing == []

    standard, missing = skills_for_role("standard", library, version="v2")
    assert [s.frontmatter.name for s in standard] == [
        "notify-user",
        "order-management-v2",
    ]
    assert missing == []


def test_unknown_role_falls_back_to_the_smallest_set():
    """権限の判断で迷ったら狭い方へ倒す。"""
    library = load_library(COMMON, DOMAIN)
    skills, _ = skills_for_role("unknown-role", library)
    assert [s.frontmatter.name for s in skills] == list(ROLE_SKILLS["free"])


def test_missing_skill_is_reported_not_silently_dropped():
    """admin が要求する admin-operations はライブラリに無い。黙って消さない。"""
    library = load_library(COMMON, DOMAIN)
    _, missing = skills_for_role("admin", library)
    assert missing == ["admin-operations"]


def test_version_selection_is_deterministic_and_splits():
    """同じ利用者には常に同じ版を返し、全体ではおおよそ半々に分かれる。"""
    assert select_version("user-1") == select_version("user-1")
    picks = [select_version(f"user-{i}") for i in range(200)]
    assert set(picks) == {"v1", "v2"}
    v1 = picks.count("v1")
    assert 60 <= v1 <= 140, f"偏りが大きい: v1={v1}/200"
    with pytest.raises(ValueError):
        select_version("user-1", versions=())


def test_resolve_names_falls_back_to_the_logical_name():
    library = load_library(COMMON, DOMAIN)
    assert resolve_names(["order-management", "notify-user"], library, "v1") == [
        "order-management-v1",
        "notify-user",
    ]


def test_tool_references_are_checked_mechanically():
    """教材は機械的に検出できないと書くが、本文の参照は拾える。"""
    library = load_library(COMMON, DOMAIN)
    v2 = library["order-management-v2"]
    assert referenced_tools(v2) == {"get_order_status", "cancel_order"}
    assert missing_tools(v2, ["get_order_status"]) == {"cancel_order"}
    assert missing_tools(v2, ["get_order_status", "cancel_order"]) == set()


class ReadonlyContextStub:
    """SkillToolset が読むのは agent_name と state と invocation_id だけ。"""

    def __init__(self, agent_name: str, state: dict) -> None:
        self.agent_name = agent_name
        self.state = state
        self.invocation_id = "invocation-1"


async def test_additional_tools_appear_only_after_the_skill_is_activated():
    """metadata.adk_additional_tools は、有効化されたスキルの分だけツールを足す。

    教材の条件付き読み込みは Toolset を作り分ける手だが、
    ADK には State を見て公開範囲を変えるこの口がある。
    """

    def cancel_order(order_id: str) -> dict:
        """注文を取り消す。

        Args:
            order_id: 注文 ID。例: ORD-12345
        """
        return {"status": "ok", "order_id": order_id}

    library = load_library(COMMON, DOMAIN)
    toolset = SkillToolset(
        [library["order-management-v2"]], additional_tools=[cancel_order]
    )
    try:
        base = {tool.name for tool in await toolset.get_tools()}
        assert "cancel_order" not in base

        activated = ReadonlyContextStub(
            "support_agent",
            {"_adk_activated_skill_support_agent": ["order-management-v2"]},
        )
        after = {tool.name for tool in await toolset.get_tools(activated)}
        assert "cancel_order" in after
    finally:
        await toolset.close()
