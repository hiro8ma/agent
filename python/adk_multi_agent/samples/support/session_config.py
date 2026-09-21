"""環境ごとに Session と Memory の保管先を選ぶ。

    AGENT_ENV=dev      InMemorySessionService（プロセスが終われば消える）
    AGENT_ENV=staging  DatabaseSessionService（SESSION_DB_URL の DB に残る）
    AGENT_ENV=prod     VertexAiSessionService（Agent Engine に残る）

必須の環境変数が無ければ、起動時に ValueError で止める。
既定値で黙って InMemory に倒すと、本番で会話が消えるまで気づかない。
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from google.adk.memory import (
    BaseMemoryService,
    InMemoryMemoryService,
    VertexAiMemoryBankService,
)
from google.adk.sessions import (
    BaseSessionService,
    DatabaseSessionService,
    InMemorySessionService,
    VertexAiSessionService,
)

ENVIRONMENTS = ("dev", "staging", "prod")
VERTEX_KEYS = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "AGENT_ENGINE_ID")


def agent_env(env: Mapping[str, str] = os.environ) -> str:
    value = env.get("AGENT_ENV", "dev")
    if value not in ENVIRONMENTS:
        raise ValueError(
            f"AGENT_ENV は {'/'.join(ENVIRONMENTS)} のいずれか（{value!r}）"
        )
    return value


def require(env: Mapping[str, str], *keys: str) -> list[str]:
    """全ての鍵の値を返す。欠けていれば、欠けた鍵を全部挙げて止める。"""
    missing = [k for k in keys if not env.get(k)]
    if missing:
        raise ValueError(f"環境変数が未設定: {', '.join(missing)}")
    return [env[k] for k in keys]


def create_session_service(env: Mapping[str, str] = os.environ) -> BaseSessionService:
    match agent_env(env):
        case "dev":
            return InMemorySessionService()
        case "staging":
            (url,) = require(env, "SESSION_DB_URL")
            return DatabaseSessionService(db_url=url)
        case _:
            project, location, engine = require(env, *VERTEX_KEYS)
            return VertexAiSessionService(
                project=project, location=location, agent_engine_id=engine
            )


def memory_enabled(env: Mapping[str, str] = os.environ) -> bool:
    return env.get("ENABLE_MEMORY_BANK", "").lower() == "true"


def create_memory_service(
    env: Mapping[str, str] = os.environ,
) -> BaseMemoryService | None:
    """Memory を使わないなら None。dev はメモリ内、それ以外は Memory Bank。

    InMemoryMemoryService は英字の語でしか一致しないので、日本語の対話では何も思い出さない。
    dev で Memory の動きを確かめるなら、英字を含む発話にする。
    """
    if not memory_enabled(env):
        return None
    if agent_env(env) == "dev":
        return InMemoryMemoryService()
    project, location, engine = require(env, *VERTEX_KEYS)
    return VertexAiMemoryBankService(
        project=project, location=location, agent_engine_id=engine
    )
