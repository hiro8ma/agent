from __future__ import annotations

from corpus import TRAINING_ARTICLES, documents, load_kenpou, training_text


def test_kenpou_fixture_has_every_article() -> None:
    articles = load_kenpou()

    assert len(articles) == 103
    assert articles[0].startswith("第一条")
    assert articles[-1].startswith("第百三条")


def test_training_text_ends_where_the_heldout_document_starts() -> None:
    articles = load_kenpou()
    trained, heldout = documents()[:2]

    assert training_text() == trained.text
    assert training_text().endswith(articles[TRAINING_ARTICLES - 1])
    assert heldout.text.startswith("第八十一条")
