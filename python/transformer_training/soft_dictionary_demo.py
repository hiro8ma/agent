"""Attention = ソフトディクショナリの直感デモ（映画推薦）

通常の dict はキー完全一致で 1 つの値を返す（ハード辞書）。Attention は query と
全 key の類似度で全 value を重み付き平均する（ソフト辞書）。
  key   = 映画の特徴 (action, drama, comedy)
  value = 評価点
  query = 新作の特徴 → 近い映画ほど高い重みで評価をブレンドして予測

実装は本リポの行列版 Attention と同一の 3 ステップ:
  1. 類似度  key @ query        (model.py の q @ k^T に対応)
  2. 重み    softmax(sim / T)   (01_softmax + temperature_demo に対応)
  3. 出力    weights @ values   (05_attention の attn @ v に対応)
温度 T を上げると分布が平滑化する（temperature_demo と同じ挙動）。
"""

from __future__ import annotations

import numpy as np

# key=映画の特徴(action, drama, comedy) / value=評価点(0-100)
MOVIE_PREFERENCES: dict[tuple[int, int, int], int] = {
    (8, 2, 3): 85,
    (3, 9, 1): 70,
    (1, 2, 9): 60,
    (5, 5, 5): 75,
    (7, 6, 2): 80,
    (2, 7, 6): 65,
    (9, 1, 1): 90,
}


def soft_dictionary(
    query: tuple[int, ...],
    table: dict[tuple[int, int, int], int],
    temperature: float = 1.0,
) -> tuple[float, np.ndarray]:
    keys = np.array(list(table), dtype=float)
    vals = np.array(list(table.values()), dtype=float)
    sims = keys @ np.array(query, dtype=float)          # 内積=類似度
    weights = np.exp(sims / max(temperature, 1e-8))     # softmax(温度付き)
    weights /= weights.sum()
    return float(weights @ vals), weights               # 重み付き和


if __name__ == "__main__":
    new_movie = (6, 4, 5)
    print(f"=== Attention = ソフトディクショナリ（query={new_movie}） ===\n")
    for t in (1.0, 5.0, 20.0):
        pred, w = soft_dictionary(new_movie, MOVIE_PREFERENCES, t)
        top = max(zip(MOVIE_PREFERENCES, w), key=lambda kv: kv[1])
        print(f"T={t:>4}: 予測 {pred:5.2f}点  最大重み {top[0]} = {top[1] * 100:4.1f}%")

    print("\nT=1 の各映画の重み:")
    _, w = soft_dictionary(new_movie, MOVIE_PREFERENCES)
    for k, wi in zip(MOVIE_PREFERENCES, w):
        print(f"  {k}: {wi * 100:5.2f}% {'#' * int(wi * 40)}")

    print("\n観察ポイント")
    print("  - query に近い特徴の映画が高い重み、遠い映画はほぼ 0 = ソフトな lookup")
    print("  - T を上げると重みが平滑化（全映画を均等に混ぜる方向）")
    print("  - 3 ステップは行列版 Attention（04/05, model.py）と同一構造")
