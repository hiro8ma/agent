"""旅行プランナーの検索ツール。

本番では Places API や経路検索 API を呼ぶが、ここは固定データを返す。

戻り値は常に list[dict] に揃える。
教材の指示は「戻り値は list[dict]」と「エラーは辞書で返す」の両方だが、
そのまま書くと成功時と失敗時で型が変わる。ツールの宣言は型ヒントから
作られるので、ここでは失敗時も {"error": ...} を 1 件だけ含むリストで返す。

例外は投げない。投げるとモデルは何が起きたか分からず、同じ呼び出しを繰り返す。
"""

from __future__ import annotations

from google.adk.tools import FunctionTool

_SPOTS: dict[str, list[dict]] = {
    "京都": [
        {
            "name": "金閣寺",
            "category": "歴史",
            "duration_hours": 1.5,
            "entry_fee_yen": 500,
        },
        {
            "name": "伏見稲荷大社",
            "category": "歴史",
            "duration_hours": 2.0,
            "entry_fee_yen": 0,
        },
        {
            "name": "嵐山竹林",
            "category": "自然",
            "duration_hours": 2.0,
            "entry_fee_yen": 0,
        },
        {
            "name": "京都国立博物館",
            "category": "アート",
            "duration_hours": 2.5,
            "entry_fee_yen": 700,
        },
    ],
    "沖縄": [
        {
            "name": "美ら海水族館",
            "category": "自然",
            "duration_hours": 3.0,
            "entry_fee_yen": 2180,
        },
        {
            "name": "首里城",
            "category": "歴史",
            "duration_hours": 1.5,
            "entry_fee_yen": 400,
        },
        {
            "name": "古宇利島",
            "category": "自然",
            "duration_hours": 3.0,
            "entry_fee_yen": 0,
        },
    ],
    "金沢": [
        {
            "name": "兼六園",
            "category": "自然",
            "duration_hours": 1.5,
            "entry_fee_yen": 320,
        },
        {
            "name": "ひがし茶屋街",
            "category": "歴史",
            "duration_hours": 1.5,
            "entry_fee_yen": 0,
        },
        {
            "name": "21世紀美術館",
            "category": "アート",
            "duration_hours": 2.0,
            "entry_fee_yen": 450,
        },
    ],
}

_RESTAURANTS: dict[str, list[dict]] = {
    "京都": [
        {"name": "瓢亭", "cuisine": "和食", "budget_per_person_yen": 15000},
        {
            "name": "イル・ギオットーネ",
            "cuisine": "イタリアン",
            "budget_per_person_yen": 8000,
        },
        {"name": "% ARABICA", "cuisine": "カフェ", "budget_per_person_yen": 1000},
        {"name": "権太呂", "cuisine": "和食", "budget_per_person_yen": 2500},
    ],
    "沖縄": [
        {"name": "首里そば", "cuisine": "和食", "budget_per_person_yen": 900},
        {
            "name": "ジャッキーステーキハウス",
            "cuisine": "洋食",
            "budget_per_person_yen": 3500,
        },
        {"name": "浜辺の茶屋", "cuisine": "カフェ", "budget_per_person_yen": 1200},
    ],
    "金沢": [
        {"name": "近江町市場 海鮮丼", "cuisine": "和食", "budget_per_person_yen": 3000},
        {
            "name": "respiración",
            "cuisine": "スペイン料理",
            "budget_per_person_yen": 9000,
        },
        {"name": "東出珈琲店", "cuisine": "カフェ", "budget_per_person_yen": 800},
    ],
}

_TRANSPORT: dict[tuple[str, str], list[dict]] = {
    ("東京", "京都"): [
        {"mode": "新幹線", "duration_minutes": 135, "price_yen": 13320},
        {"mode": "飛行機", "duration_minutes": 70, "price_yen": 15000},
        {"mode": "高速バス", "duration_minutes": 420, "price_yen": 4000},
    ],
    ("東京", "沖縄"): [
        {"mode": "飛行機", "duration_minutes": 165, "price_yen": 25000},
    ],
    ("東京", "金沢"): [
        {"mode": "新幹線", "duration_minutes": 150, "price_yen": 14180},
        {"mode": "飛行機", "duration_minutes": 60, "price_yen": 17000},
    ],
    ("大阪", "京都"): [
        {"mode": "JR新快速", "duration_minutes": 29, "price_yen": 580},
        {"mode": "新幹線", "duration_minutes": 15, "price_yen": 1450},
    ],
}


def search_tourist_spots(city: str, interests: list[str]) -> list[dict]:
    """指定された都市の観光スポットを検索する。

    Args:
        city: 都市名。例: 京都
        interests: 興味のあるカテゴリ。例: ["歴史", "自然"]。空リストなら絞り込まない

    Returns:
        観光スポットの一覧。各要素は name / category / duration_hours / entry_fee_yen を持つ。
        見つからないときは error を 1 件だけ含むリストを返す
    """
    spots = _SPOTS.get(city.strip())
    if not spots:
        return [{"error": f"{city} の観光スポットは登録されていません"}]
    if interests:
        matched = [s for s in spots if s["category"] in interests]
        if matched:
            return matched
    return spots


def search_restaurants(city: str, cuisine: str, budget_per_person: int) -> list[dict]:
    """指定された都市のレストランを予算内で検索する。

    Args:
        city: 都市名。例: 京都
        cuisine: 料理の種類。例: 和食。空文字なら絞り込まない
        budget_per_person: 1 人あたりの上限。単位は円。例: 5000。0 以下なら上限なし

    Returns:
        レストランの一覧。各要素は name / cuisine / budget_per_person_yen を持つ。
        見つからないときは error を 1 件だけ含むリストを返す
    """
    shops = _RESTAURANTS.get(city.strip())
    if not shops:
        return [{"error": f"{city} のレストランは登録されていません"}]
    matched = shops
    if cuisine:
        matched = [s for s in matched if cuisine in s["cuisine"]]
    if budget_per_person > 0:
        matched = [
            s for s in matched if s["budget_per_person_yen"] <= budget_per_person
        ]
    if not matched:
        return [
            {
                "error": f"{city} に条件（{cuisine} / {budget_per_person} 円以内）に合う店がありません"
            }
        ]
    return matched


def search_transport(origin: str, destination: str, date: str) -> list[dict]:
    """出発地から目的地までの交通手段を検索する。

    Args:
        origin: 出発地。例: 東京
        destination: 目的地。例: 京都
        date: 利用日。ISO 8601 の日付。例: 2026-10-03

    Returns:
        交通手段の一覧。各要素は mode / duration_minutes / price_yen を持つ。
        見つからないときは error を 1 件だけ含むリストを返す
    """
    options = _TRANSPORT.get((origin.strip(), destination.strip()))
    if not options:
        return [{"error": f"{origin} から {destination} の経路は登録されていません"}]
    return [dict(o, date=date) for o in options]


spot_search_tool = FunctionTool(func=search_tourist_spots)
restaurant_search_tool = FunctionTool(func=search_restaurants)
transport_search_tool = FunctionTool(func=search_transport)
