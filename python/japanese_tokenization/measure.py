from __future__ import annotations

import codecs
from collections.abc import Iterable
from dataclasses import dataclass

from adapters import TokenizerAdapter

_VISIBLE = {" ": "␣", "\n": "\\n", "\t": "\\t"}


@dataclass(frozen=True)
class Count:
    chars: int
    utf8_bytes: int
    tokens: int
    split_chars: int

    @property
    def tokens_per_char(self) -> float:
        return self.tokens / self.chars

    @property
    def split_char_ratio(self) -> float:
        return self.split_chars / self.chars


def count_split_characters(token_bytes: Iterable[bytes]) -> int:
    decoder = codecs.getincrementaldecoder("utf-8")()
    split = 0
    pending = False
    for data in token_bytes:
        decoded = decoder.decode(data)
        if pending and decoded:
            split += 1
        pending = bool(decoder.getstate()[0])
    return split


def count(tokenizer: TokenizerAdapter, text: str) -> Count:
    ids = tokenizer.encode(text)
    return Count(
        chars=len(text),
        utf8_bytes=len(text.encode("utf-8")),
        tokens=len(ids),
        split_chars=count_split_characters(tokenizer.token_bytes(i) for i in ids),
    )


def render_bytes(data: bytes) -> str:
    parts: list[str] = []
    hex_run: list[str] = []
    i = 0
    while i < len(data):
        char = ""
        for size in range(1, 5):
            try:
                char = data[i : i + size].decode("utf-8")
            except UnicodeDecodeError:
                continue
            break
        if not char:
            hex_run.append(f"{data[i]:02x}")
            i += 1
            continue
        if hex_run:
            parts.append("<" + " ".join(hex_run) + ">")
            hex_run = []
        parts.append("".join(_VISIBLE.get(c, c) for c in char))
        i += len(char.encode("utf-8"))
    if hex_run:
        parts.append("<" + " ".join(hex_run) + ">")
    return "".join(parts)


def render_ids(tokenizer: TokenizerAdapter, ids: list[int]) -> list[str]:
    return [render_bytes(tokenizer.token_bytes(i)) for i in ids]


def pieces(tokenizer: TokenizerAdapter, text: str) -> list[str]:
    return render_ids(tokenizer, tokenizer.encode(text))


def roundtrips(tokenizer: TokenizerAdapter, text: str) -> bool:
    return tokenizer.decode(tokenizer.encode(text)) == text
