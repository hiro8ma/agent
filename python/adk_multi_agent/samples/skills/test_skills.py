"""Agent Skills の読み込みと、ツールとしての載せ方を固定する。

スキルは SKILL.md 1 枚から成る。

    frontmatter   name と description が必須。name はディレクトリ名と一致する
    本文          instructions になる
    下位ディレクトリ  references / assets / scripts が resources になる

name とディレクトリ名がずれると読み込み時に落ちる。
配布物として配るときに、名前だけ書き換えて壊す事故を防いでいる。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from google.adk.skills import list_skills_in_dir, load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from pydantic import ValidationError

SKILLS_DIR = Path(__file__).parent
SKILL_DIR = SKILLS_DIR / "travel-budget"


def test_skill_loads_from_dir():
    skill = load_skill_from_dir(SKILL_DIR)
    assert skill.frontmatter.name == "travel-budget"
    assert "見積" in skill.frontmatter.description
    assert "交通費を往復で数える" in skill.instructions
    assert skill.frontmatter.license == "Apache-2.0"


def test_directory_name_must_match_the_frontmatter_name(tmp_path: Path):
    """名前がずれたら読み込み時に落ちる。"""
    renamed = tmp_path / "budget-of-travel"
    shutil.copytree(SKILL_DIR, renamed)
    with pytest.raises(ValueError, match="does not match directory"):
        load_skill_from_dir(renamed)


def test_listing_skills_returns_frontmatter_only():
    """一覧は frontmatter だけを返す。本文を読まずに候補を絞れる。"""
    listed = list_skills_in_dir(SKILLS_DIR)
    assert "travel-budget" in listed
    assert listed["travel-budget"].description


async def test_skill_toolset_exposes_four_tools():
    """スキル 1 つにつき本文が載るのではなく、扱うためのツールが生える。

    一覧は frontmatter だけを返すので、本文を読まずに候補を絞れる。
    常時すべての本文を載せる必要がない点で、コンテキスト予算にも効く。
    """
    skill = load_skill_from_dir(SKILL_DIR)
    toolset = SkillToolset([skill])
    try:
        tools = await toolset.get_tools()
        assert sorted(tool.name for tool in tools) == [
            "list_skills",
            "load_skill",
            "load_skill_resource",
            "run_skill_script",
        ]
    finally:
        await toolset.close()


def test_skill_name_must_be_kebab_case(tmp_path: Path):
    """name は小文字のケバブケースに限られる。教材に無い制約。"""
    bad = tmp_path / "travel_budget"
    shutil.copytree(SKILL_DIR, bad)
    md = bad / "SKILL.md"
    md.write_text(
        md.read_text(encoding="utf-8").replace(
            "name: travel-budget", "name: travel_budget"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="kebab-case"):
        load_skill_from_dir(bad)


def test_duplicate_skill_names_are_rejected():
    """同じ名前を 2 つ載せると組み立て時に落ちる。"""
    skill = load_skill_from_dir(SKILL_DIR)
    with pytest.raises(ValueError, match="Duplicate skill name"):
        SkillToolset([skill, skill])
