"""State のキー設計と、型を確かめてから読む入口。

State は `dict[str, Any]` なので、型も鍵の綴りも保証されない。
そこで読み書きの入口を 1 か所へ集める。

    StateKeys        鍵の定数。綴り違いをコードの側で潰す
    get_user_tier    値域を確かめてから返す
    get_max_results  型と範囲を確かめてから返す
    UserState        まとまった単位を Pydantic で受ける
    branch_key       並列で走る枝ごとに鍵を分ける
    looks_sensitive  State へ置いてはいけない鍵を弾く

鍵のプレフィックスは寿命を決める。
`app:` はアプリ全体、`user:` は同じ利用者、無印はその Session、`temp:` はその Invocation。
`temp:` だけは保存されない。実行中は読めるが、イベントに乗る手前で落とされる。
"""

from __future__ import annotations

from typing import Any

from google.adk.sessions.state import State
from pydantic import BaseModel, Field

# 値域。文字列を直接書くと、判定とモデル定義がずれていく。
TIERS = ("free", "standard", "premium")

# 検索件数の下限と上限。設定ミスで 0 件や 10 万件を返さないようにする。
MIN_RESULTS = 1
MAX_RESULTS = 100
DEFAULT_RESULTS = 5


class StateKeys:
    """State の鍵。プレフィックスまで含めて定数にする。

    プレフィックスを定数の中に畳み込むと、呼び出し側から寿命が読めなくなる。
    `StateKeys.USER_TIER` を見れば `user:` が付いていると分かる形にしておく。
    """

    APP_VERSION = "app:version"
    APP_MAX_RESULTS = "app:max_search_results"

    USER_NAME = "user:name"
    USER_TIER = "user:tier"
    USER_LANGUAGE = "user:preferred_language"

    TEMP_SEARCH_COUNT = "temp:search_count"
    TEMP_CURRENT_INTENT = "temp:current_intent"


def scope_of(key: str) -> str:
    """鍵からスコープを返す。プレフィックスが無ければ session。"""
    if key.startswith(State.APP_PREFIX):
        return "app"
    if key.startswith(State.USER_PREFIX):
        return "user"
    if key.startswith(State.TEMP_PREFIX):
        return "temp"
    return "session"


def get_user_tier(state: Any) -> str:
    """会員区分を返す。想定外の値は free へ倒す。

    State には利用者の入力や過去の版の値が残りうる。
    知らない値をそのまま下流へ流すと、権限の判定が通ってしまう。
    """
    tier = state.get(StateKeys.USER_TIER, "free")
    if tier not in TIERS:
        return "free"
    return tier


def get_max_results(state: Any) -> int:
    """検索件数の上限を返す。型と範囲を確かめる。

    設定値は人が書くので、文字列や桁違いの数が入る。
    読む側で丸めておくと、ツールの実装から検証が消える。
    """
    raw = state.get(StateKeys.APP_MAX_RESULTS, DEFAULT_RESULTS)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return DEFAULT_RESULTS
    return max(MIN_RESULTS, min(MAX_RESULTS, raw))


class UserState(BaseModel):
    """利用者に紐づく値をまとめて受ける。"""

    name: str = "ゲスト"
    tier: str = Field(default="free", pattern="^(free|standard|premium)$")
    preferred_language: str = "ja"


def load_user_state(state: Any) -> UserState:
    """State から利用者の情報を組み立てる。

    不正な値は Pydantic が弾くが、ここで弾くと対話そのものが落ちる。
    読み出しは落とさない方針にして、値域を外れたものは既定値へ倒す。
    """
    tier = get_user_tier(state)
    return UserState(
        name=state.get(StateKeys.USER_NAME, "ゲスト"),
        tier=tier,
        preferred_language=state.get(StateKeys.USER_LANGUAGE, "ja"),
    )


def save_user_state(state: Any, user: UserState) -> None:
    """利用者の情報を State へ書き戻す。

    書き込みは実行中の context.state に対して行う。
    SessionService から取得した Session の state は素の dict で、
    そこへ代入してもイベントに乗らず保存されない。
    """
    state[StateKeys.USER_NAME] = user.name
    state[StateKeys.USER_TIER] = user.tier
    state[StateKeys.USER_LANGUAGE] = user.preferred_language


def branch_key(base: str, branch: str) -> str:
    """並列で走る枝ごとに鍵を分ける。

    同じ鍵へ複数のエージェントが書くと、後に終わった方が勝ち、
    先の値は最終状態から消える。例外も警告も出ない。
    イベントの state_delta には両方残るので、後から追うことはできる。
    """
    return f"{base}_{branch}"


# State へ置くべきでない鍵の語。ログや開発 UI に出る前提で考える。
_SENSITIVE_WORDS = ("token", "password", "secret", "api_key", "credential")


def looks_sensitive(key: str) -> bool:
    """機密らしい鍵かを返す。

    State は開発 UI にもログにも出る。
    認証情報は環境変数か Secret Manager に置き、State へは識別子だけを持つ。
    """
    lowered = key.lower()
    return any(word in lowered for word in _SENSITIVE_WORDS)
