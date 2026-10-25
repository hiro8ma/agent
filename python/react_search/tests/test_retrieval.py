from __future__ import annotations

import pytest

from retrieval import (
    BM25,
    article_numbers,
    bigrams,
    containment,
    load_articles,
    substring_search,
)

ARTICLES = load_articles()
INDEX = BM25(ARTICLES)


def test_load_articles_numbers_from_one() -> None:
    assert len(ARTICLES) == 103
    assert ARTICLES[0].label == "第一条"
    assert ARTICLES[66].number == 67
    assert ARTICLES[66].label == "第六十七条"


def test_bigrams_split_on_punctuation() -> None:
    assert bigrams("国会、天") == ["国会", "天"]


@pytest.mark.parametrize(
    ("query", "want"),
    [
        pytest.param("国会の議決で指名", 67, id="prime minister nomination"),
        pytest.param("参議院議員の任期", 46, id="upper house term"),
        pytest.param("弾劾裁判所", 64, id="impeachment court"),
        pytest.param("憲法 改正 発議", 96, id="amendment"),
    ],
)
def test_bm25_returns_expected_article_in_top3(query: str, want: int) -> None:
    got = INDEX.search(query)
    assert len(got) == 3
    assert want in [a.number for a in got]


@pytest.mark.parametrize(
    ("query", "want"),
    [
        pytest.param("弾劾裁判所", [64], id="single term"),
        pytest.param("緊急集会 臨時会", [53, 54], id="any term counts"),
        pytest.param("首相", [], id="word not in text"),
    ],
)
def test_substring_search_returns_articles_containing_terms(query: str, want: list[int]) -> None:
    assert sorted(a.number for a in substring_search(ARTICLES, query)) == want


def test_bm25_misses_word_absent_from_corpus() -> None:
    assert INDEX.search("首相") == []


def test_containment_is_share_of_part_found_in_whole() -> None:
    assert containment("国会議員", "国会議員の任期") == 1.0
    assert containment("宇宙", "国会議員の任期") == 0.0
    assert containment("", "国会") == 0.0


@pytest.mark.parametrize(
    ("text", "want"),
    [
        pytest.param("第十条", {10}, id="ten"),
        pytest.param("第六十七条", {67}, id="tens and ones"),
        pytest.param("第百三条", {103}, id="hundred"),
        pytest.param("第113条と第９条", {113, 9}, id="arabic and full width digits"),
        pytest.param("条文なし", set(), id="none"),
    ],
)
def test_article_numbers_parses_kanji_and_digits(text: str, want: set[int]) -> None:
    assert article_numbers(text) == want


def test_article_labels_round_trip() -> None:
    assert all(article_numbers(a.label) == {a.number} for a in ARTICLES)
