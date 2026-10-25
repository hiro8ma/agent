from __future__ import annotations

import importlib.util
import io
import json
import re
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Protocol

import sentencepiece as spm
import tiktoken
from tokenizers import Tokenizer

REFERENCE_BPE_PATH = Path(__file__).resolve().parent.parent / "transformer_training" / "bpe.py"
_BYTE_PIECE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


class TokenizerAdapter(Protocol):
    name: str
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def token_bytes(self, token_id: int) -> bytes: ...


@cache
def _byte_level_alphabet() -> dict[str, int]:
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    byte_values = list(printable)
    code_points = list(printable)
    shift = 0
    for b in range(256):
        if b not in printable:
            byte_values.append(b)
            code_points.append(256 + shift)
            shift += 1
    return {chr(c): b for b, c in zip(byte_values, code_points, strict=True)}


def byte_level_token_to_bytes(token: str) -> bytes:
    # GPT-2 系のバイト単位 BPE は語彙を表示できる文字に置き換えて持つので、元のバイトに戻す
    alphabet = _byte_level_alphabet()
    return bytes(alphabet[ch] for ch in token)


def metaspace_piece_to_bytes(piece: str) -> bytes:
    match = _BYTE_PIECE.fullmatch(piece)
    if match:
        return bytes([int(match.group(1), 16)])
    return piece.replace("▁", " ").encode("utf-8")


class TiktokenAdapter:
    def __init__(self, encoding_name: str) -> None:
        self._enc = tiktoken.get_encoding(encoding_name)
        self.name = encoding_name
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    def token_bytes(self, token_id: int) -> bytes:
        return self._enc.decode_single_token_bytes(token_id)


class HuggingFaceAdapter:
    def __init__(self, repo_id: str, name: str) -> None:
        self._tok = Tokenizer.from_pretrained(repo_id)
        self.name = name
        self.vocab_size = self._tok.get_vocab_size()
        decoder = json.loads(self._tok.to_str()).get("decoder")
        self._byte_level = "ByteLevel" in json.dumps(decoder)

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def token_bytes(self, token_id: int) -> bytes:
        token = self._tok.id_to_token(token_id)
        if token is None:
            raise ValueError(f"unknown token id {token_id}")
        if self._byte_level:
            return byte_level_token_to_bytes(token)
        return metaspace_piece_to_bytes(token)


@cache
def _reference_bpe_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("transformer_training_bpe", REFERENCE_BPE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {REFERENCE_BPE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ByteBPEAdapter:
    def __init__(self, training_text: str, vocab_size: int, name: str) -> None:
        self._tok = _reference_bpe_module().BPETokenizer()
        self._tok.train(training_text, vocab_size)
        self.name = name
        self.vocab_size = int(self._tok.vocab_size)
        self._bytes = {i: bytes([i]) for i in range(256)}
        for (left, right), new_id in self._tok.merges:
            self._bytes[new_id] = self._bytes[left] + self._bytes[right]

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self._tok.encode(text, allow_special=False)]

    def decode(self, ids: list[int]) -> str:
        return str(self._tok.decode(ids))

    def token_bytes(self, token_id: int) -> bytes:
        return self._bytes[token_id]


class SentencePieceAdapter:
    def __init__(self, lines: Iterable[str], vocab_size: int, model_type: str, name: str) -> None:
        model = io.BytesIO()
        # NFKC や空白の詰めは decode で元の文字列に戻らなくなるので切る
        # 学習コーパスに空白が無いと空白の印が語彙に入らず、バイトに落ちて空白に戻らないので足す
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(lines),
            model_writer=model,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            byte_fallback=True,
            user_defined_symbols=["▁"],
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            bos_id=-1,
            eos_id=-1,
            num_threads=1,
            minloglevel=2,
        )
        self._sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
        self.name = name
        self.vocab_size = int(self._sp.get_piece_size())

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self._sp.encode(text)]

    def decode(self, ids: list[int]) -> str:
        return str(self._sp.decode(ids))

    def token_bytes(self, token_id: int) -> bytes:
        return metaspace_piece_to_bytes(str(self._sp.id_to_piece(token_id)))

    def vocabulary(self) -> set[str]:
        return {
            str(self._sp.id_to_piece(i))
            for i in range(self.vocab_size)
            if not (self._sp.is_byte(i) or self._sp.is_unknown(i))
        }

    def nbest(self, text: str, size: int) -> list[list[int]]:
        return [[int(i) for i in ids] for ids in self._sp.nbest_encode_as_ids(text, size)]
