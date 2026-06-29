"""単語頻度つきの文字ベース BPE デモ。

`bpe.py` は GPT 系に近い byte-level BPE 実装。このファイルは教材でよく出る
「単語頻度を数える -> 単語を文字へ分割 -> 最頻隣接ペアを merge」の最小版。

目的は実用 tokenizer ではなく、BPE training loop の構造を目で追えるようにすること。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

Corpus = list[str]
Pair = tuple[str, str]
Splits = dict[str, list[str]]
WordFreqs = dict[str, int]
Merges = list[Pair]


DEMO_CORPUS: Corpus = [
    "Large language models are transforming the landscape of natural language processing.",
    "Tokenization improves language models.",
    "Subword tokenization helps language models handle unknown words.",
    "Byte pair encoding creates useful subword tokens.",
]


def simple_tokenize(text: str) -> list[str]:
    """空白だけで分割する最小 tokenizer。

    句読点は単語に付いたまま残る。これは教材の最初のステップを再現するため。
    """
    return text.split()


def build_word_freqs(corpus: Iterable[str]) -> WordFreqs:
    """corpus 全体の単語頻度を数える。"""
    freqs: defaultdict[str, int] = defaultdict(int)
    for text in corpus:
        for word in simple_tokenize(text):
            freqs[word] += 1
    return dict(freqs)


def build_initial_vocab(word_freqs: WordFreqs) -> list[str]:
    """単語に含まれる文字を集めて初期 vocabulary を作る。"""
    return sorted({char for word in word_freqs for char in word})


def split_words(word_freqs: WordFreqs) -> Splits:
    """各単語を文字 token の列へ分割する。"""
    return {word: list(word) for word in word_freqs}


def compute_pair_freqs(splits: Splits, word_freqs: WordFreqs) -> Counter[Pair]:
    """隣接 token ペアの頻度を、単語頻度で重みづけして数える。"""
    pair_freqs: Counter[Pair] = Counter()
    for word, split in splits.items():
        word_freq = word_freqs[word]
        for pair in zip(split, split[1:]):
            pair_freqs[pair] += word_freq
    return pair_freqs


def merge_pair(pair: Pair, splits: Splits) -> Splits:
    """全単語内の pair を結合した新しい splits を返す。"""
    left, right = pair
    merged_token = left + right
    next_splits: Splits = {}
    for word, split in splits.items():
        merged: list[str] = []
        i = 0
        while i < len(split):
            if i < len(split) - 1 and split[i] == left and split[i + 1] == right:
                merged.append(merged_token)
                i += 2
            else:
                merged.append(split[i])
                i += 1
        next_splits[word] = merged
    return next_splits


def train_word_bpe(corpus: Iterable[str], num_merges: int) -> tuple[list[str], Merges, Splits]:
    """文字ベース BPE を num_merges 回だけ学習する。"""
    word_freqs = build_word_freqs(corpus)
    vocab = build_initial_vocab(word_freqs)
    splits = split_words(word_freqs)
    merges: Merges = []

    for _ in range(num_merges):
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        if not pair_freqs:
            break
        # 頻度 -> 辞書順で決定的に選ぶ。教材理解では再現性が大事。
        best_pair = max(pair_freqs, key=lambda pair: (pair_freqs[pair], pair))
        merges.append(best_pair)
        vocab.append(best_pair[0] + best_pair[1])
        splits = merge_pair(best_pair, splits)

    return vocab, merges, splits


def tokenize_word(word: str, merges: Merges) -> list[str]:
    """学習済み merge ルールを順に適用して 1 単語を subword 化する。"""
    split = list(word)
    for pair in merges:
        split = merge_pair(pair, {word: split})[word]
    return split


def tokenize_text(text: str, merges: Merges) -> list[str]:
    """空白 tokenize 後、各単語に BPE merge を適用する。"""
    out: list[str] = []
    for word in simple_tokenize(text):
        out.extend(tokenize_word(word, merges))
    return out


def main() -> None:
    print("=== 単語頻度つき文字ベース BPE デモ ===\n")

    word_freqs = build_word_freqs(DEMO_CORPUS)
    print("[1] word frequencies")
    for word, freq in sorted(word_freqs.items(), key=lambda item: (-item[1], item[0]))[:8]:
        print(f"  {word:16} {freq}")

    initial_vocab = build_initial_vocab(word_freqs)
    print(f"\n[2] initial vocab ({len(initial_vocab)} chars)")
    print("  " + " ".join(initial_vocab))

    vocab, merges, splits = train_word_bpe(DEMO_CORPUS, num_merges=10)
    print("\n[3] learned merges")
    for i, pair in enumerate(merges, start=1):
        print(f"  {i:02d}: {pair!r} -> {pair[0] + pair[1]!r}")

    print(f"\n[4] final vocab size = {len(vocab)}")
    print("  added tokens: " + ", ".join(vocab[len(initial_vocab) :]))

    for word in ("language", "Tokenization", "processing.", "unknown"):
        print(f"\n[5] {word!r} -> {tokenize_word(word, merges)}")

    sample = "Tokenization improves LLMs."
    print(f"\n[6] sample: {sample!r}")
    print(f"  tokens: {tokenize_text(sample, merges)}")

    print("\n[7] corpus splits after training")
    for word in ("language", "models", "tokenization", "subword"):
        if word in splits:
            print(f"  {word:12} -> {splits[word]}")


if __name__ == "__main__":
    main()
