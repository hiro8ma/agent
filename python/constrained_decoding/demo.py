"""Offline demo: how logits masking changes the output distribution.

実モデルも GPU も使わない。固定のロジット（対数確率）を用意し、
LogitsProcessor 風のマスクを適用して softmax / サンプリングがどう変わるかを対比する。

教材の 3 要点を再現する:
  (1) beam search 的に複数候補を見る -> ここでは top-k 表示で分布を見せる
  (2) 入力はトークン ID なので decode してルール適用 -> AllowedTokens で history を decode して判定
  (3) ロジットは対数確率なので確率ゼロ = -inf 代入 -> softmax 後に 0.000 になることを確認
  (4) スタイル制御の best-of -> hard ban(-inf) は分布が NaN 化、soft mask(-10000) は最良を 1 本残す
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from processors import (
    AllowedTokensLogitsProcessor,
    BannedTokensLogitsProcessor,
    LogitsProcessorList,
    ScoredBestOfLogitsProcessor,
    softmax,
)

# --- 極小トークナイザ（id <-> token 文字列） ---------------------------------
VOCAB = ["{", "}", '"name"', '"age"', ":", ",", "alice", "42", "BAD", "  "]
ID_TO_TOK = dict(enumerate(VOCAB))
TOK_TO_ID = {t: i for i, t in ID_TO_TOK.items()}


def decode(ids: list[int]) -> str:
    return "".join(ID_TO_TOK.get(i, "?") for i in ids)


def show_distribution(label: str, scores: np.ndarray, top: int = 5) -> None:
    probs = softmax(scores)[0]
    order = np.argsort(probs)[::-1][:top]
    print(f"  [{label}]")
    for idx in order:
        bar = "#" * int(round(probs[idx] * 40))
        token = ID_TO_TOK[int(idx)]
        print(f"    {token:>8}  p={probs[idx]:.3f}  {bar}")
    print()


def main() -> None:
    rng = np.random.default_rng(0)

    # 固定ロジット（対数確率）。BAD トークンに高いスコアを持たせて、マスクの効果を見せる。
    base_scores = np.array(
        [[1.0, 0.2, 2.0, 1.5, 0.1, 0.1, 1.8, 1.7, 3.0, -1.0]],
        dtype=np.float64,
    )
    # これまでに生成済みのトークン列（バッチ 1 件）。decode してルール適用に使う。
    input_ids = np.array([[TOK_TO_ID["{"], TOK_TO_ID['"name"'], TOK_TO_ID[":"]]])

    print(f"history (decoded): {decode(input_ids[0].tolist())!r}\n")

    # (c) マスク無し: BAD が最有力
    print("=== (1) マスク無し ===")
    show_distribution("raw logits -> softmax", base_scores)

    # (a) 禁止語マスク: BAD と空白を物理的に禁止
    print("=== (2) 禁止語マスク（BAD / 空白を -inf） ===")
    banned = BannedTokensLogitsProcessor([TOK_TO_ID["BAD"], TOK_TO_ID["  "]])
    show_distribution("banned -> softmax", banned(input_ids, base_scores))

    # (b) allow-list マスク（JSON Schema 強制の最小版）
    # 直前トークンが ':' なら、次は「値」しか来てはいけない、という状態機械。
    def json_value_allowed(history: Sequence[int]) -> set[int]:
        last = ID_TO_TOK.get(history[-1]) if history else None
        if last == ":":
            return {TOK_TO_ID["alice"], TOK_TO_ID["42"]}  # 値だけ許可
        return set(TOK_TO_ID.values())

    print("=== (3) allow-list マスク（直前が ':' なので値トークンのみ許可） ===")
    allow = AllowedTokensLogitsProcessor(json_value_allowed)
    masked = allow(input_ids, base_scores)
    show_distribution("allow-listed -> softmax", masked)

    # (組合せ) processor チェーン + 実サンプリングで BAD が一度も出ないことを確認
    print("=== (4) processor チェーン + サンプリング 1000 回 ===")
    chain = LogitsProcessorList([banned, allow])
    probs = softmax(chain(input_ids, base_scores))[0]
    draws = rng.choice(len(VOCAB), size=1000, p=probs)
    counts = {ID_TO_TOK[i]: int((draws == i).sum()) for i in range(len(VOCAB)) if (draws == i).any()}
    print(f"  sampled tokens: {counts}")
    bad_count = int((draws == TOK_TO_ID["BAD"]).sum())
    print(f"  BAD が選ばれた回数: {bad_count}  (制約により 0 が期待値)")

    best_of_demo()


# --- スタイル制御の best-of（栄養サプリ例） ----------------------------------
STYLE_VOCAB = ["whey", "premium", "protein", "quality", "growth", "perfect", "for", "you"]
STYLE_ID_TO_TOK = dict(enumerate(STYLE_VOCAB))
STYLE_TOK_TO_ID = {t: i for i, t in STYLE_ID_TO_TOK.items()}
POSITIVE = {"whey", "premium"}
NEGATIVE = {"quality", "growth", "perfect"}


def style_decode(ids: Sequence[int]) -> str:
    return " ".join(STYLE_ID_TO_TOK.get(i, "?") for i in ids)


def style_evaluate(text: str) -> float:
    words = text.split()
    return sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)


def best_of_demo() -> None:
    print("=== (5) hard ban(-inf) vs soft best-of(-10000) の対比 ===")

    def ids(tokens: list[str]) -> list[int]:
        return [STYLE_TOK_TO_ID[t] for t in tokens]

    # 候補シーケンス（beam search の各 beam を模す）
    candidates = np.array(
        [
            ids(["premium", "whey", "protein", "for", "you"]),  # positive=2 negative=0 -> 最良
            ids(["quality", "growth", "perfect", "for", "you"]),  # positive=0 negative=3
            ids(["premium", "quality", "growth", "for", "you"]),  # positive=1 negative=2
        ]
    )
    for r in range(candidates.shape[0]):
        text = style_decode(candidates[r].tolist())
        print(f"  candidate[{r}] score={style_evaluate(text):+.0f}  {text!r}")

    # 全候補に同一の固定ロジット（行ごとの差はスコア関数だけで決める）
    vocab_size = len(STYLE_VOCAB)
    scores = np.tile(np.linspace(0.0, 1.0, vocab_size), (candidates.shape[0], 1))

    print("\n  -- hard ban (-inf) を全候補に当てたら --")
    hard = scores.copy()
    hard[1:, :] = float("-inf")  # 最良以外を -inf に
    probs_hard = softmax(hard)
    nan_rows = int(np.isnan(probs_hard).any(axis=1).sum())
    print(f"     NaN を含む行数: {nan_rows}  (-inf 行は 0/0 で分布が壊れる)")

    print("\n  -- soft best-of (-10000) なら --")
    best_of = ScoredBestOfLogitsProcessor(style_decode, style_evaluate)
    probs_soft = softmax(best_of(candidates, scores))
    nan_rows_soft = int(np.isnan(probs_soft).any(axis=1).sum())
    row_mass = probs_soft.sum(axis=1)
    print(f"     NaN を含む行数: {nan_rows_soft}  (全行が有効な分布のまま)")
    print(f"     各候補の確率質量: {np.round(row_mass, 3).tolist()}")
    kept = int(np.argmax(probs_soft.max(axis=1)))
    print(f"     最良として残った候補: index {kept}  {style_decode(candidates[kept].tolist())!r}")


if __name__ == "__main__":
    main()
