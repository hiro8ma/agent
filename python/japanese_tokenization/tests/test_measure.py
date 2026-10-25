from __future__ import annotations

import pytest

from measure import count, count_split_characters, pieces, render_bytes


class ByteTokenizer:
    name = "bytes"
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")

    def token_bytes(self, token_id: int) -> bytes:
        return bytes([token_id])


@pytest.mark.parametrize(
    ("text", "chars", "utf8_bytes"),
    [
        pytest.param("カレー", 3, 9, id="katakana is 3 bytes per char"),
        pytest.param("CPU 80%", 7, 7, id="ascii is 1 byte per char"),
        pytest.param("第一条　天皇\n", 7, 19, id="full-width space and newline count as chars"),
    ],
)
def test_count_uses_code_points_and_utf8_bytes(text: str, chars: int, utf8_bytes: int) -> None:
    result = count(ByteTokenizer(), text)

    assert result.chars == chars
    assert result.utf8_bytes == utf8_bytes
    assert result.tokens == utf8_bytes
    assert result.tokens_per_char == pytest.approx(utf8_bytes / chars)


@pytest.mark.parametrize(
    ("tokens", "want"),
    [
        pytest.param([b"\xe6\xbb\x85"], 0, id="whole char in one token"),
        pytest.param([b"\xe6\xbb", b"\x85"], 1, id="char split into two tokens"),
        pytest.param([b"\xe6", b"\xbb", b"\x85"], 1, id="char split into three tokens"),
        pytest.param(
            [b"\xe4\xbb", b"\x8f\xe6\xbb", b"\x85"], 2, id="token ends inside the next char"
        ),
        pytest.param([b"ab", b"c"], 0, id="ascii never splits"),
    ],
)
def test_count_split_characters(tokens: list[bytes], want: int) -> None:
    assert count_split_characters(tokens) == want


def test_byte_tokenizer_splits_every_non_ascii_char() -> None:
    result = count(ByteTokenizer(), "仏滅 ok")

    assert result.split_chars == 2
    assert result.split_char_ratio == pytest.approx(2 / 5)


@pytest.mark.parametrize(
    ("data", "want"),
    [
        pytest.param("滅".encode(), "滅", id="whole char"),
        pytest.param(b"\xe6\xbb", "<e6 bb>", id="leading bytes of a char"),
        pytest.param(b"\x85", "<85>", id="trailing byte of a char"),
        pytest.param("は".encode() + b"\xe4", "は<e4>", id="char followed by a partial char"),
        pytest.param(b"\xbb" + "は".encode(), "<bb>は", id="partial char followed by a char"),
        pytest.param(b" a\n", "␣a\\n", id="space and newline made visible"),
    ],
)
def test_render_bytes(data: bytes, want: str) -> None:
    assert render_bytes(data) == want


def test_pieces_show_each_byte_of_a_kanji() -> None:
    assert pieces(ByteTokenizer(), "滅") == ["<e6>", "<bb>", "<85>"]
