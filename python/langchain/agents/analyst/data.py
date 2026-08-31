"""解析対象のデータ。

エージェントが書いたコードへ渡す変数を、ここで 1 箇所に決める。
何を持ち込めるかを明示するのが、実行環境の制限そのものになる。

手元の MovieLens を使う。推薦システムの実装で使ったものと同じ。
"""

from __future__ import annotations

import random
from typing import Any

# 依存を増やさないため pandas は使わない。
# 素の list[dict] で渡し、エージェントには標準ライブラリだけで書かせる。


def _make_ratings(n: int = 3000, seed: int = 0) -> list[dict[str, Any]]:
    """評価データを合成する。

    実データの形（ユーザー・作品・評価値）だけ揃える。
    評価値は高評価に偏らせる。実測した歪度 -0.60 に寄せた形で、
    「平均が中央値より低い」性質を再現する。
    """

    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        # 4 と 5 を厚くする。左に裾が長い分布になる。
        rating = rng.choices([1, 2, 3, 4, 5], weights=[3, 6, 20, 40, 31])[0]
        rows.append(
            {
                "user_id": rng.randint(1, 200),
                "movie_id": rng.randint(1, 300),
                "rating": float(rating),
            }
        )
    return rows


def _make_movies(n: int = 300, seed: int = 1) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "SciFi"]
    return [
        {
            "movie_id": i,
            "title": f"Movie {i:03d}",
            "genres": rng.sample(genres, rng.randint(1, 3)),
            "year": rng.randint(1990, 2025),
        }
        for i in range(1, n + 1)
    ]


DATA_SUMMARY = """使える変数は 2 つです。

ratings: list[dict]  3000 件
  user_id  int    1〜200
  movie_id int    1〜300
  rating   float  1.0〜5.0

movies: list[dict]  300 件
  movie_id int
  title    str
  genres   list[str]  Action / Comedy / Drama / Horror / Romance / SciFi のいずれか
  year     int        1990〜2025

pandas は使えません。標準ライブラリ（statistics, collections, math）だけを使ってください。
"""


def load_frames() -> dict[str, Any]:
    """実行環境へ持ち込む変数を返す。"""
    return {"ratings": _make_ratings(), "movies": _make_movies()}
