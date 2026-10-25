from __future__ import annotations

import pytest

from adapters import (
    TiktokenAdapter,
    TokenizerAdapter,
    byte_level_token_to_bytes,
    metaspace_piece_to_bytes,
)
from corpus import SHORT_SENTENCES, documents
from experiment import VOCAB_SIZE, build_own

EDGE_CASES = ["a  b", "\n\n", "第一条　天皇", "😊", "ﾊﾝｶｸ"]
TEXTS = [d.text for d in documents()] + [s for _, s in SHORT_SENTENCES] + EDGE_CASES


@pytest.fixture(scope="module")
def own_tokenizers() -> list[TokenizerAdapter]:
    return list(build_own())


@pytest.fixture(scope="module")
def o200k() -> TokenizerAdapter:
    try:
        return TiktokenAdapter("o200k_base")
    except Exception as exc:
        pytest.skip(f"o200k_base を取得できない（{type(exc).__name__}）")


def test_own_tokenizers_share_the_vocab_size(own_tokenizers: list[TokenizerAdapter]) -> None:
    assert [t.vocab_size for t in own_tokenizers] == [VOCAB_SIZE] * len(own_tokenizers)


@pytest.mark.parametrize("text", TEXTS, ids=range(len(TEXTS)))
def test_own_tokenizers_roundtrip_and_tokens_cover_every_byte(
    own_tokenizers: list[TokenizerAdapter], text: str
) -> None:
    for tok in own_tokenizers:
        ids = tok.encode(text)

        assert tok.decode(ids) == text, tok.name
        assert b"".join(tok.token_bytes(i) for i in ids) == text.encode("utf-8"), tok.name


@pytest.mark.parametrize("text", TEXTS, ids=range(len(TEXTS)))
def test_tiktoken_roundtrips_and_tokens_cover_every_byte(
    o200k: TokenizerAdapter, text: str
) -> None:
    ids = o200k.encode(text)

    assert o200k.decode(ids) == text
    assert b"".join(o200k.token_bytes(i) for i in ids) == text.encode("utf-8")


@pytest.mark.parametrize(
    ("token", "want"),
    [
        pytest.param("æ»ħ", "滅".encode(), id="kanji in one token"),
        pytest.param("ä»", b"\xe4\xbb", id="leading bytes of a kanji"),
        pytest.param("ĠCPU", b" CPU", id="leading space"),
        pytest.param("Ċ", b"\n", id="newline"),
    ],
)
def test_byte_level_token_to_bytes(token: str, want: bytes) -> None:
    assert byte_level_token_to_bytes(token) == want


@pytest.mark.parametrize(
    ("piece", "want"),
    [
        pytest.param("<0xE6>", b"\xe6", id="byte fallback"),
        pytest.param("▁仏", " 仏".encode(), id="space marker"),
        pytest.param("には", "には".encode(), id="plain piece"),
    ],
)
def test_metaspace_piece_to_bytes(piece: str, want: bytes) -> None:
    assert metaspace_piece_to_bytes(piece) == want
