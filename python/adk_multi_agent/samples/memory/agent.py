"""Session を越えて覚えるエージェントと、State のスコープ判定。

3 層は寿命とスコープで分かれる。

    Session   1 回の対話。イベント列と State を持つ
    State     Session の中の Key-Value。接頭辞でスコープが決まる
    Memory    Session を越える。MemoryService 経由で読み書きする

保存の経路は 1 本しかない。
`session.state[key] = value` と書いても捨てられる。
`Session.state` の型は素の dict で、SessionService が返すのはコピーだからである。
残るのは、実行中の `context.state`（State 型）へ書いた値だけで、
これがイベントの `state_delta` に乗って初めて保存される。

Memory は保存と検索の両方を自分で書く。
Runner に memory_service を渡すだけでは何も溜まらない。

    保存   after_agent_callback で `await ctx.add_session_to_memory()`
    検索   PreloadMemoryTool（毎回自動）か LoadMemoryTool（モデルが呼ぶ）
"""

from __future__ import annotations

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.sessions.state import State
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

MODEL = "gemini-3.8-flash"

INSTRUCTION = """あなたは利用者の好みを覚える案内役です。
過去の会話が必要な場合は、与えられた記憶だけを使って答えてください。
記憶に無いことを推測して答えないでください。"""


def scope_of(key: str) -> str:
    """State の鍵からスコープを返す。

    接頭辞が無い鍵は session スコープになる。
    ADK 側に定数があるのは 3 つだけで、session は「どれでもない」で表される。
    """
    if key.startswith(State.APP_PREFIX):
        return "app"
    if key.startswith(State.USER_PREFIX):
        return "user"
    if key.startswith(State.TEMP_PREFIX):
        return "temp"
    return "session"


def is_persisted(key: str) -> bool:
    """その鍵が保存されるかを返す。

    temp: だけが保存されない。
    実行中は読めるので「使えない」わけではなく、イベントに乗る手前で落とされる。
    """
    return scope_of(key) != "temp"


def visible_across_sessions(key: str) -> bool:
    """同じ利用者の別 Session から見えるかを返す。"""
    return scope_of(key) in ("app", "user")


async def save_session_to_memory(callback_context: CallbackContext) -> None:
    """対話の終わりに、この Session を Memory へ取り込む。

    memory_service を渡していない Runner では ValueError が上がり、
    実行そのものが失敗する。握り潰されないので、設定漏れは実行時に分かる。
    """
    await callback_context.add_session_to_memory()


def build_agent(model: object = MODEL) -> Agent:
    """記憶する案内役を組み立てる。

    PreloadMemoryTool はモデルが呼ぶツールではない。
    毎リクエストで直前の利用者発話を検索し、結果を instruction へ足す。
    取りこぼしは無い代わりに、毎回トークンを使う。
    モデルに選ばせたい場合は LoadMemoryTool を渡す。
    """
    return Agent(
        name="memory_agent",
        model=model,
        description="過去の対話を覚えて次の対話で使う案内役",
        instruction=INSTRUCTION,
        tools=[PreloadMemoryTool()],
        after_agent_callback=save_session_to_memory,
    )


root_agent = build_agent()
