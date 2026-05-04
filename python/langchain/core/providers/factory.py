from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from .openai import build_openai_chat_model

SUPPORTED_PROVIDERS = ("openai",)


def select_provider() -> BaseChatModel:
    """Pick a chat model based on env vars.

    LLM_PROVIDER (default 'openai') decides the backend.
    LLM_MODEL overrides the per-provider default model.
    Unsupported providers raise immediately so typos do not silently fall back.
    """

    provider = (os.environ.get("LLM_PROVIDER") or "openai").lower()

    if provider == "openai":
        return build_openai_chat_model()

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER: {provider!r}. "
        f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}."
    )
