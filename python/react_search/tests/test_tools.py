from __future__ import annotations

import pytest

from retrieval import BM25, load_articles
from tools import Action, Environment, first_finish, parse_action, stop_strings, truncate_at_stop


@pytest.mark.parametrize(
    ("text", "want"),
    [
        pytest.param(
            " 探す\n行動 1: search[内閣総理大臣 指名]",
            Action("search", "内閣総理大臣 指名"),
            id="search",
        ),
        pytest.param("行動 2: lookup[任期]", Action("lookup", "任期"), id="lookup"),
        pytest.param("行動 3: finish[国会]", Action("finish", "国会"), id="finish"),
        pytest.param(
            "行動 1：Search［改正］", Action("search", "改正"), id="full-width colon and brackets"
        ),
        pytest.param(
            "行動 1: search[a]\n行動 2: finish[b]", Action("search", "a"), id="first action wins"
        ),
    ],
)
def test_parse_action_accepts_known_tools(text: str, want: Action) -> None:
    assert parse_action(text) == want


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("search[天皇]", id="no action label"),
        pytest.param("行動 1: google[天皇]", id="unknown tool"),
        pytest.param("行動 1: search 天皇", id="no brackets"),
        pytest.param("行動 1: search[天皇", id="unclosed bracket"),
        pytest.param("行動 1: finish[]", id="empty argument"),
        pytest.param("行動 1: search[天皇\n]", id="argument across lines"),
    ],
)
def test_parse_action_rejects_malformed(text: str) -> None:
    assert parse_action(text) is None


@pytest.mark.parametrize(
    ("text", "want", "stopped"),
    [
        pytest.param(
            " a\n行動 1: search[x]\n観測 1: 作文\n思考 2: b",
            " a\n行動 1: search[x]",
            True,
            id="newline label",
        ),
        pytest.param(
            " a 行動 1: search[x] 観測 1: 作文", " a 行動 1: search[x] ", True, id="inline label"
        ),
        pytest.param(" a\n行動 1: finish[y]", " a\n行動 1: finish[y]", False, id="no stop"),
    ],
)
def test_truncate_at_stop(text: str, want: str, stopped: bool) -> None:
    assert truncate_at_stop(text, stop_strings(1)) == (want, stopped)


def test_truncate_at_stop_uses_earliest_stop() -> None:
    assert truncate_at_stop("abcXdefY", ["Y", "X"]) == ("abc", True)


def test_first_finish_skips_earlier_search() -> None:
    text = " a\n行動 1: search[x]\n観測 1: 作文\n思考 2: b\n行動 2: finish[答え]"
    assert first_finish(text) == Action("finish", "答え")


def test_lookup_returns_sentences_of_opened_articles() -> None:
    articles = load_articles()
    env = Environment(BM25(articles).search, articles)
    assert "先に search" in env.lookup("任期")
    env.search("衆議院議員の任期")
    got = env.lookup("任期")
    assert "第四十五条　衆議院議員の任期は、四年とする。" in got.splitlines()
    assert env.searches == 1
    assert env.lookups == 2
    assert "見つからなかった" in env.lookup("宇宙")
