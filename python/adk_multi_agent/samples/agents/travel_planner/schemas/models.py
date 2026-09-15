"""旅行プランの出力スキーマ。

最終出力の形を Pydantic で固定する。Field の description は
モデルへ渡る仕様書なので、全フィールドに付ける。

入れ子は 3 層まで。TravelPlan → DaySchedule → TouristSpot が上限で、
これ以上深くすると、モデルが形を外したときの原因が追えなくなる。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TouristSpot(BaseModel):
    """観光スポット 1 件"""

    name: str = Field(description="スポット名。例: 金閣寺")
    category: str = Field(description="カテゴリ。例: 歴史、自然、アート")
    duration_hours: float = Field(description="滞在の目安。単位は時間。例: 1.5")
    entry_fee_yen: int = Field(default=0, description="入場料。単位は円。無料なら 0")


class Restaurant(BaseModel):
    """食事 1 件"""

    name: str = Field(description="店名。例: 瓢亭")
    cuisine: str = Field(description="料理の種類。例: 和食、イタリアン")
    meal: str = Field(description="どの食事か。例: 朝食、昼食、夕食")
    budget_per_person_yen: int = Field(description="1 人あたりの目安。単位は円")


class TransportOption(BaseModel):
    """交通手段 1 件"""

    mode: str = Field(description="手段。例: 新幹線、飛行機、高速バス")
    duration_minutes: int = Field(description="所要時間。単位は分")
    price_yen: int = Field(description="片道の料金。単位は円")


class DaySchedule(BaseModel):
    """1 日ぶんの日程"""

    day: int = Field(description="何日目か。1 から始める")
    spots: list[TouristSpot] = Field(description="その日に回る観光スポット")
    restaurants: list[Restaurant] = Field(description="その日の食事")
    notes: str = Field(default="", description="移動や注意点のメモ")


class BudgetBreakdown(BaseModel):
    """予算の内訳。単位はすべて円"""

    transport_yen: int = Field(description="交通費。往復と現地移動の合計")
    food_yen: int = Field(description="食費。日数ぶんの合計")
    activity_yen: int = Field(description="入場料とアクティビティ費の合計")
    total_yen: int = Field(description="合計。3 項目の和と一致させる")


class TravelPlan(BaseModel):
    """旅行プランの最終出力"""

    destination: str = Field(description="旅行先。例: 京都")
    duration_days: int = Field(description="旅行日数。例: 2")
    schedule: list[DaySchedule] = Field(description="日程ごとのスケジュール")
    transport: list[TransportOption] = Field(description="往復に使う交通手段の候補")
    budget: BudgetBreakdown = Field(description="予算の内訳")
    highlights: list[str] = Field(description="このプランの見どころ。3 件程度")
    tips: list[str] = Field(default_factory=list, description="持ち物や注意点")
