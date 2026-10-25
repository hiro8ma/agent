from __future__ import annotations

import json
from pathlib import Path

from llm import MODEL, Generation, MLXModel, Request, cache_key


def test_cached_generation_is_returned_without_loading_model(tmp_path: Path) -> None:
    request = Request("問い", "思考 1:")
    key = cache_key(MODEL, request, 160, ("\n観測",))
    path = tmp_path / "generations.jsonl"
    path.write_text(
        json.dumps({"key": key, "text": " 探す", "tokens": 2, "seconds": 0.5}) + "\n",
        encoding="utf-8",
    )
    model = MLXModel(path)
    assert model.generate([request], 160, ("\n観測",)) == [Generation(" 探す", 2, 0.5)]
    assert model._loaded is None
    assert model.generated_seconds == 0.0


def test_cache_key_depends_on_prefix_and_stop() -> None:
    base = cache_key(MODEL, Request("問い", "思考 1:"), 160, ("\n観測",))
    assert base != cache_key(MODEL, Request("問い", "思考 2:"), 160, ("\n観測",))
    assert base != cache_key(MODEL, Request("問い", "思考 1:"), 160, ())
