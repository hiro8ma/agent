"""Logits processors that physically forbid tokens by writing -inf into scores.

これは Hugging Face Transformers の ``LogitsProcessor`` と同じ契約に形を合わせている:

    __call__(input_ids, scores) -> scores

- ``input_ids``: これまで生成したトークン ID 列（バッチ x 系列長）
- ``scores``:    語彙全体に対する次トークンのロジット（対数確率、バッチ x 語彙数）

ロジットは softmax を通す前の対数確率なので、``-inf`` を代入したトークンは
softmax 後に確率がちょうど 0 になり、サンプリング / beam search で物理的に選べなくなる。
"これが本番の Outlines / llama.cpp grammar / vLLM guided decoding の最小核" である:
それらは「次に来てよいトークン集合」を文法 / JSON Schema / 正規表現から動的に計算し、
集合外のロジットを -inf にして無効化する。本デモはその allow-list 計算と -inf 代入だけを
numpy で取り出したもの。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

import numpy as np

NEG_INF = -np.inf


class LogitsProcessor(Protocol):
    """HF Transformers の LogitsProcessor と同じ呼び出し契約。"""

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray: ...


class BannedTokensLogitsProcessor:
    """指定 token id のロジットを -inf にして出力不能にする（禁止語マスク）。

    HF の ``NoBadWordsLogitsProcessor`` の 1 トークン版に相当する。
    """

    def __init__(self, banned_token_ids: Iterable[int]) -> None:
        self.banned_token_ids = sorted(set(banned_token_ids))

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        scores = scores.copy()
        if self.banned_token_ids:
            scores[:, self.banned_token_ids] = NEG_INF
        return scores


class AllowedTokensLogitsProcessor:
    """許可集合だけを通し、それ以外を -inf にする（allow-list マスク）。

    許可集合は固定にも、これまでのトークン列に応じて動的にも決められる。
    動的版が JSON Schema / 文法強制の最小モデル: 「今の状態で次に来てよいトークン」だけ残す。

    Parameters
    ----------
    allowed_fn:
        ``input_ids`` の 1 系列（1 次元配列）を受け取り、許可 token id の集合を返す関数。
    """

    def __init__(self, allowed_fn: Callable[[Sequence[int]], Iterable[int]]) -> None:
        self.allowed_fn = allowed_fn

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        scores = scores.copy()
        vocab_size = scores.shape[1]
        for row in range(scores.shape[0]):
            history = input_ids[row].tolist()
            allowed = set(self.allowed_fn(history))
            mask = np.ones(vocab_size, dtype=bool)
            for tok in allowed:
                if 0 <= tok < vocab_size:
                    mask[tok] = False
            scores[row, mask] = NEG_INF
        return scores


class LogitsProcessorList:
    """複数 processor を順に適用する（HF の同名クラスと同契約）。"""

    def __init__(self, processors: Iterable[LogitsProcessor] | None = None) -> None:
        self.processors: list[LogitsProcessor] = list(processors or [])

    def append(self, processor: LogitsProcessor) -> None:
        self.processors.append(processor)

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        for proc in self.processors:
            scores = proc(input_ids, scores)
        return scores


SOFT_MASK = -10000.0


class ScoredBestOfLogitsProcessor:
    """候補シーケンスをスコア関数で採点し、最良以外を soft mask で下げる（スタイル制御）。

    各行（候補）の input_ids を ``decode_fn`` で文字列に戻し ``score_fn`` で採点。
    最大スコア以外の行のロジット全体を -10000 にする（-inf ではない）。

    理由: 全候補を -inf にすると行全体が確率ゼロになり softmax が 0/0 で NaN 化し分布が壊れる。
    best-of では soft mask（-10000）を使い、最良候補を必ず 1 本残す。
    """

    def __init__(
        self,
        decode_fn: Callable[[Sequence[int]], str],
        score_fn: Callable[[str], float],
    ) -> None:
        self.decode_fn = decode_fn
        self.score_fn = score_fn

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        scores = scores.copy()
        row_scores = [self.score_fn(self.decode_fn(input_ids[r].tolist())) for r in range(scores.shape[0])]
        best = int(np.argmax(row_scores))
        for row in range(scores.shape[0]):
            if row != best:
                scores[row, :] = SOFT_MASK
        return scores


def softmax(scores: np.ndarray) -> np.ndarray:
    """数値安定版 softmax。-inf 入力は確率 0 になる。"""
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(shifted)  # exp(-inf) = 0
    result: np.ndarray = exp / np.sum(exp, axis=-1, keepdims=True)
    return result
