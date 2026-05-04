from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024


def build_openai_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Construct a ChatOpenAI instance.

    Resolution order: explicit arg > env var > module default.
    Raises a RuntimeError if OPENAI_API_KEY is missing so misconfiguration
    surfaces before the first network call.
    """

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running the agent."
        )

    resolved_model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
    resolved_temperature = (
        temperature
        if temperature is not None
        else _float_env("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    )
    resolved_max_tokens = (
        max_tokens
        if max_tokens is not None
        else _int_env("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    )

    return ChatOpenAI(
        model=resolved_model,
        temperature=resolved_temperature,
        max_completion_tokens=resolved_max_tokens,
    )


def _float_env(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float for env {key}: {raw!r}") from exc


def _int_env(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid int for env {key}: {raw!r}") from exc
