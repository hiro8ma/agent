"""解析対象のデータと、その概要をコンテキストへ渡す仕組み。

手書きの説明文をプロンプトに埋めていたが、それでは実データとずれる。
列を足しても説明は変わらず、エージェントは存在しない列を参照する。
**概要はデータから生成する。**

教材と同じく `df.info()` / `df.sample(5)` / `df.describe()` の 3 点を渡す。
それぞれ役割が違う。

  info      列名と型と欠損の有無。**どの列が使えるか**
  sample    実際の値。**値の形（日付の書式、カテゴリの文字列）**
  describe  分布。**外れ値や偏りの見当**
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "campaign.csv"
TEMPLATE_FILE = Path(__file__).parent / "prompts" / "describe_dataframe.jinja"


def load_template(template_file: Path) -> Template:
    env = Environment(loader=FileSystemLoader(template_file.parent))
    return env.get_template(template_file.name)


def generate_dataset(rows: int = 800, seed: int = 0) -> pd.DataFrame:
    """マーケティング施策の消費者行動データを合成する。

    教材のスキーマに合わせる。実データは手元に無いが、
    列の構成と型が同じなら、エージェントの書くコードは同じ形になる。
    """

    rng = random.Random(seed)
    channels = ["SNS", "メール", "検索エンジン"]
    types = ["割引", "ポイント付与", "送料無料"]
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    times = ["朝", "昼", "夕方", "夜", "深夜"]

    records = []
    for i in range(rows):
        clicks = rng.randint(1, 400)
        # 購入はクリックの一部。コンバージョン率が 1 を超えないようにする。
        purchases = rng.randint(0, max(1, clicks // rng.randint(3, 40)))
        start = f"2026-{rng.randint(1, 8):02d}-{rng.randint(1, 28):02d}"
        records.append(
            {
                "campaign_id": f"CMP-{i // 8 + 1:04d}",
                "channel_id": rng.choice(channels),
                "campaign_type": rng.choice(types),
                "start_date": start,
                "end_date": f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}",
                "login_at": f"{start} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00",
                "login_interval": round(rng.uniform(0.5, 30.0), 2),
                "purchase_date": start if purchases else None,
                "purchase_amount": round(rng.uniform(500, 80000), 0) if purchases else None,
                "purchase_items": purchases if purchases else None,
                "click_date": start,
                "click_count": clicks,
                "conversion_rate": round(purchases / clicks, 4),
                "score": round(rng.gauss(60, 18), 1),
                "is_successful": purchases > 0 and rng.random() < 0.6,
                "weekday": rng.choice(weekdays),
                "time_of_day": rng.choice(times),
                "week_of_month": rng.randint(1, 5),
            }
        )
    return pd.DataFrame(records)


def ensure_dataset() -> Path:
    """データファイルが無ければ作る。"""
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(DATA_FILE, index=False)
    return DATA_FILE


def describe_dataframe(data_file: Path | None = None) -> str:
    """データ概要を文字列で返す。プロンプトへ埋める。"""

    path = data_file or ensure_dataset()
    df = pd.read_csv(path)

    buf = io.StringIO()
    df.info(buf=buf)

    return load_template(TEMPLATE_FILE).render(
        df_info=buf.getvalue(),
        df_sample=df.sample(5, random_state=0).to_string(),
        df_describe=df.describe().to_string(),
    )


def load_frames() -> dict:
    """実行環境へ持ち込む変数を返す。

    **モジュールは入れない。**
    別プロセスへ送るものは直列化できる必要があり、
    モジュールを渡すと `cannot pickle 'module' object` で落ちる。
    ライブラリは実行側のコードで import させる。

    DataFrame は pickle できるので、そのまま渡せる。
    リモート実行（E2B）に切り替える場合は JSON にする必要があるため、
    CSV のパスを渡して向こうで読ませる形になる。
    """

    # _csv_path はリモート実行でファイルを送るための経路。
    # ローカル実行では使わないが、同じ関数で両方に対応させる。
    path = ensure_dataset()
    return {"df": pd.read_csv(path), "_csv_path": str(path)}
