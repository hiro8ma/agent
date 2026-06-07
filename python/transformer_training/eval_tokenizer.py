"""Tokenizer の圧縮率評価

学習ノートの観点を再現する評価ユーティリティ。

圧縮率（compression ratio）= byte 数 / token 数 = 1 トークンあたりの平均バイト数。
値が大きいほど列が短くなり、同じ context 長でより多くの情報を詰め込める。

トレードオフ:
  - 語彙サイズ ↑  → 圧縮率 ↑（頻出パターンをまとめられる）
  - 語彙サイズ ↑  → 埋め込み層パラメータ ↑（vocab_size * n_embd が線形に増える）

CharTokenizer（文字単位）と BPE（バイト起点 + マージ）を同じテキストで比較し、
さらに target_vocab_size を変えて圧縮率がどう動くかを表で観察する。
"""

from __future__ import annotations

from typing import Protocol

from bpe import BPETokenizer
from data import CharTokenizer, download_dataset


class _SupportsEncode(Protocol):
    """encode を持つ tokenizer のための構造的型（CharTokenizer / BPETokenizer 両対応）"""

    def encode(self, text: str) -> list[int]: ...


def compression_ratio(tokenizer: _SupportsEncode, text: str) -> float:
    """byte 数 / token 数（1 トークンあたり平均バイト数）を返す

    byte 数は UTF-8 換算で数える。CharTokenizer は 1 文字 1 token なので
    マルチバイト文字があると 1 を超えるが、tinyshakespeare は ASCII 主体なので
    ほぼ 1.0 になる。BPE はマージにより 1 を大きく超える。
    """
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tokenizer.encode(text))
    if n_tokens == 0:
        return 0.0
    return n_bytes / n_tokens


def _report_line(name: str, n_bytes: int, n_tokens: int, ratio: float) -> str:
    return (
        f"  {name:18} byte {n_bytes:>8,}  token {n_tokens:>8,}  "
        f"圧縮率 {ratio:5.2f} 倍"
    )


# === 圧縮率の評価 ===
# 直接実行: uv run python eval_tokenizer.py
if __name__ == "__main__":
    SAMPLE_CHARS = 50_000  # 学習が数秒で終わる範囲のサンプル

    print("=== Tokenizer 圧縮率の評価 ===\n")

    full_text = download_dataset()
    sample = full_text[:SAMPLE_CHARS]
    sample_bytes = len(sample.encode("utf-8"))
    print(f"評価サンプル: tinyshakespeare 先頭 {SAMPLE_CHARS:,} 文字"
          f"（{sample_bytes:,} byte）\n")

    print("--- (a) CharTokenizer vs (b) BPE ---")

    char_tok = CharTokenizer(full_text)  # vocab は全体から構築（未知文字を出さない）
    char_tokens = char_tok.encode(sample)
    char_ratio = compression_ratio(char_tok, sample)
    print(_report_line(
        f"Char(vocab={char_tok.vocab_size})", sample_bytes, len(char_tokens), char_ratio
    ))

    bpe = BPETokenizer()
    bpe.train(sample, target_vocab_size=500)
    bpe_tokens = bpe.encode(sample)
    bpe_ratio = compression_ratio(bpe, sample)
    print(_report_line(
        f"BPE(vocab={bpe.vocab_size})", sample_bytes, len(bpe_tokens), bpe_ratio
    ))

    if char_ratio > 0:
        print(f"\n  → BPE は CharTokenizer の {bpe_ratio / char_ratio:.2f} 倍まで列を短縮"
              "（同じ context 長で扱える情報量が増える）")

    print("\n--- 語彙サイズ ↔ 圧縮率のトレードオフ ---")
    print("  vocab を大きくすると圧縮率は上がるが、埋め込み層 (vocab_size * n_embd) も増える\n")

    n_embd = 128  # train.py のデフォルトと同じ
    header = (
        f"  {'target_vocab':>12}  {'実 vocab':>8}  {'token 数':>9}  "
        f"{'圧縮率':>7}  {'埋め込み params':>14}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for target in (300, 500, 1000):
        tok = BPETokenizer()
        tok.train(sample, target_vocab_size=target)
        ratio = compression_ratio(tok, sample)
        n_tokens = len(tok.encode(sample))
        embd_params = tok.vocab_size * n_embd
        print(
            f"  {target:>12}  {tok.vocab_size:>8}  {n_tokens:>9,}  "
            f"{ratio:>6.2f}倍  {embd_params:>14,}"
        )

    print("\n=== 観察ポイント ===")
    print("  - CharTokenizer は ASCII 中心のため圧縮率がほぼ 1.0（1 文字 = 1 token）")
    print("  - BPE は頻出ペアをまとめるので 1 token あたりのバイト数が増える")
    print("  - target_vocab_size を上げると圧縮率は上がるが、埋め込み params は線形に増える")
    print("  - 圧縮率と埋め込みコストのバランスで実用 vocab（GPT-2 は 50,257）が決まる")
