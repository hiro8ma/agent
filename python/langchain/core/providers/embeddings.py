from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def build_openai_embeddings(model: str | None = None) -> Embeddings:
    """Construct an OpenAIEmbeddings instance.

    Resolution order: explicit arg > EMBEDDING_MODEL env var > module default.
    Raises a RuntimeError if OPENAI_API_KEY is missing so misconfiguration
    surfaces before the first network call.
    """

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running the agent."
        )

    resolved_model = model or os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    return OpenAIEmbeddings(model=resolved_model)


DEFAULT_FASTEMBED_MODEL = "intfloat/multilingual-e5-large"


def build_fastembed_embeddings(model: str | None = None) -> Embeddings:
    """Construct a local, key-free FastEmbed (ONNX) embeddings instance.

    Resolution order: explicit arg > EMBEDDING_MODEL env var > module default.
    Runs fully offline after the first model download, so it needs no API key.
    """

    from langchain_community.embeddings import FastEmbedEmbeddings

    resolved_model = (
        model or os.environ.get("EMBEDDING_MODEL") or DEFAULT_FASTEMBED_MODEL
    )
    return FastEmbedEmbeddings(model_name=resolved_model)


def select_embeddings() -> Embeddings:
    """Pick an embeddings backend based on env vars.

    EMBEDDING_PROVIDER (default 'openai') decides the backend.
    EMBEDDING_MODEL overrides the per-provider default model.
    'fastembed' is the key-free local backend; 'openai' needs OPENAI_API_KEY.
    """

    provider = (os.environ.get("EMBEDDING_PROVIDER") or "openai").lower()

    if provider == "fastembed":
        return build_fastembed_embeddings()
    if provider == "openai":
        return build_openai_embeddings()

    raise RuntimeError(
        f"Unsupported EMBEDDING_PROVIDER: {provider!r}. Supported: fastembed, openai."
    )
