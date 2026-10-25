from __future__ import annotations

import platform
import re
import sys
import time
import unicodedata
from collections.abc import Callable
from importlib.metadata import version

from adapters import (
    ByteBPEAdapter,
    HuggingFaceAdapter,
    SentencePieceAdapter,
    TiktokenAdapter,
    TokenizerAdapter,
)
from corpus import SHORT_SENTENCES, Document, documents, training_lines, training_text
from measure import Count, count, pieces, render_ids, roundtrips

VOCAB_SIZE = 1000
HUGGING_FACE_TOKENIZERS = [
    ("Qwen/Qwen3-0.6B", "Qwen3"),
    ("llm-jp/llm-jp-3-1.8b", "llm-jp-3"),
]
EXAMPLES = 6
MIN_EXAMPLE_CHARS = 6
NBEST = 3
NAME_WIDTH = 28
KATAKANA_RUN = re.compile(r"[ァ-ー]+")


def build_public() -> list[TokenizerAdapter]:
    built: list[TokenizerAdapter] = [TiktokenAdapter("o200k_base"), TiktokenAdapter("cl100k_base")]
    for repo_id, name in HUGGING_FACE_TOKENIZERS:
        try:
            built.append(HuggingFaceAdapter(repo_id, name))
        except Exception as exc:
            print(f"- {repo_id} を取得できなかったので外した（{type(exc).__name__}）")
    return built


def build_own() -> tuple[ByteBPEAdapter, SentencePieceAdapter, SentencePieceAdapter]:
    return (
        ByteBPEAdapter(training_text(), VOCAB_SIZE, "自作 BPE（バイト単位）"),
        SentencePieceAdapter(training_lines(), VOCAB_SIZE, "bpe", "SentencePiece BPE"),
        SentencePieceAdapter(training_lines(), VOCAB_SIZE, "unigram", "SentencePiece ユニグラム"),
    )


def print_table(
    tokenizers: list[TokenizerAdapter],
    docs: list[Document],
    counts: dict[tuple[str, str], Count],
    cell: Callable[[Count], str],
) -> None:
    print("| トークナイザ | 語彙数 | " + " | ".join(d.name for d in docs) + " |")
    print("|---|---:|" + "---:|" * len(docs))
    for tok in tokenizers:
        cells = [cell(counts[(tok.name, d.name)]) for d in docs]
        print(f"| {tok.name} | {tok.vocab_size:,} | " + " | ".join(cells) + " |")
    print()


def pad(text: str, width: int) -> str:
    used = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - used, 1)


def clauses(text: str) -> list[str]:
    return [c for c in re.split(r"[、。\n　]", text) if c]


def main() -> None:
    started = time.perf_counter()
    print("## 実行環境\n")
    print(f"- Python {sys.version.split()[0]}、{platform.platform()}")
    libs = ["tiktoken", "tokenizers", "sentencepiece", "regex"]
    print("- " + "、".join(f"{lib} {version(lib)}" for lib in libs))

    public = build_public()
    byte_bpe, sp_bpe, sp_unigram = build_own()
    own: list[TokenizerAdapter] = [byte_bpe, sp_bpe, sp_unigram]
    tokenizers = public + own
    docs = documents()

    texts = [d.text for d in docs] + [s for _, s in SHORT_SENTENCES]
    broken = [t.name for t in tokenizers if not all(roundtrips(t, text) for text in texts)]
    if broken:
        print(f"- encode と decode の往復で元に戻らなかったトークナイザ {'、'.join(broken)}\n")
    else:
        print("- どのトークナイザも、すべての文書と短い文で encode と decode の往復で元に戻った\n")

    counts = {(t.name, d.name): count(t, d.text) for t in tokenizers for d in docs}

    print("## 文書の大きさ\n")
    print("| 文書 | 文字数 | UTF-8 のバイト数 | 1文字あたりのバイト数 |")
    print("|---|---:|---:|---:|")
    for d in docs:
        c = counts[(tokenizers[0].name, d.name)]
        print(f"| {d.name} | {c.chars:,} | {c.utf8_bytes:,} | {c.utf8_bytes / c.chars:.2f} |")
    print()

    print("## トークン数\n")
    print_table(tokenizers, docs, counts, lambda c: f"{c.tokens:,}")
    print("## 1文字あたりのトークン数\n")
    print_table(tokenizers, docs, counts, lambda c: f"{c.tokens_per_char:.2f}")
    print("## 複数のトークンに割れた文字の割合\n")
    print_table(tokenizers, docs, counts, lambda c: f"{c.split_char_ratio:.1%}")

    print("## 短い文のトークン数\n")
    print("| 文 | 文字数 | " + " | ".join(t.name for t in tokenizers) + " |")
    print("|---|---:|" + "---:|" * len(tokenizers))
    for _, sentence in SHORT_SENTENCES:
        cells = [str(len(t.encode(sentence))) for t in tokenizers]
        print(f"| {sentence} | {len(sentence)} | " + " | ".join(cells) + " |")
    print()

    print("## 短い文の分割\n")
    print("```text")
    for kind, sentence in SHORT_SENTENCES:
        print(f"{sentence}（{kind}）")
        for tok in tokenizers:
            print(f"  {pad(tok.name, NAME_WIDTH)}{' / '.join(pieces(tok, sentence))}")
        print()
    print("```\n")

    bpe_vocab, unigram_vocab = sp_bpe.vocabulary(), sp_unigram.vocabulary()
    print(f"## 自作の BPE とユニグラムの語彙（語彙数{VOCAB_SIZE:,}）\n")
    print(f"- 両方に入っている部分語は{len(bpe_vocab & unigram_vocab):,}個")
    for name, only in [
        (sp_bpe.name, bpe_vocab - unigram_vocab),
        (sp_unigram.name, unigram_vocab - bpe_vocab),
    ]:
        longest = sorted(only, key=lambda p: (-len(p), p))[: EXAMPLES * 2]
        print(f"- {name} だけにある{len(only):,}個のうち長いもの {' / '.join(longest)}")
    print()

    tech = docs[3]
    katakana = KATAKANA_RUN.findall(tech.text)
    rest = [r for r in KATAKANA_RUN.split(tech.text) if r]
    print(f"## 自作の3つで{tech.name}をカタカナとそれ以外に分けたトークン数\n")
    print(f"| トークナイザ | カタカナ（{sum(map(len, katakana))}文字） | それ以外 | 全体 |")
    print("|---|---:|---:|---:|")
    for tok in own:
        on_katakana = sum(len(tok.encode(r)) for r in katakana)
        on_rest = sum(len(tok.encode(r)) for r in rest)
        print(f"| {tok.name} | {on_katakana} | {on_rest} | {len(tok.encode(tech.text))} |")
    print()

    heldout = docs[1]
    split = [(c, pieces(sp_bpe, c), pieces(sp_unigram, c)) for c in clauses(heldout.text)]
    differ = [(c, b, u) for c, b, u in split if b != u]
    print(f"## 自作の BPE とユニグラムで分割が違う句（{heldout.name}）\n")
    print(
        f"- 読点、句点、改行、全角空白で区切った{len(split)}句のうち{len(differ)}句で分割が違った\n"
    )
    examples = [d for d in differ if len(d[0]) >= MIN_EXAMPLE_CHARS][:EXAMPLES]
    print("```text")
    for clause, bpe_pieces, unigram_pieces in examples:
        print(clause)
        print(f"  {pad(byte_bpe.name, NAME_WIDTH)}{' / '.join(pieces(byte_bpe, clause))}")
        print(f"  {pad(sp_bpe.name, NAME_WIDTH)}{' / '.join(bpe_pieces)}")
        print(f"  {pad(sp_unigram.name, NAME_WIDTH)}{' / '.join(unigram_pieces)}")
        print()
    print("```\n")

    shortest_first = sorted((c for c, _, _ in differ if len(c) >= MIN_EXAMPLE_CHARS), key=len)
    candidates = [(c, sp_unigram.nbest(c, NBEST)) for c in shortest_first]
    target = next(((c, n) for c, n in candidates if len(n) >= NBEST), None)
    if target:
        print("## ユニグラムの分割の候補\n")
        print(f"- 句「{target[0]}」の分割を、尤度の高い順に{NBEST}件並べる\n")
        print("```text")
        for rank, ids in enumerate(target[1], start=1):
            print(f"  {rank}. {' / '.join(render_ids(sp_unigram, ids))}")
        print("```\n")

    print(f"実行時間{time.perf_counter() - started:.1f}秒")


if __name__ == "__main__":
    main()
