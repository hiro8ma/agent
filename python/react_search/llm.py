from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

MODEL = "mlx-community/gemma-3-4b-it-4bit"


@dataclass(frozen=True)
class Request:
    user: str
    prefix: str = ""


@dataclass(frozen=True)
class Generation:
    text: str
    tokens: int
    seconds: float


class LLM(Protocol):
    def generate(
        self, requests: Sequence[Request], max_tokens: int, early_stop: Sequence[str] = ()
    ) -> list[Generation]: ...


def cache_key(model: str, request: Request, max_tokens: int, early_stop: Sequence[str]) -> str:
    payload = json.dumps(
        [model, request.user, request.prefix, max_tokens, list(early_stop)], ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class MLXModel:
    def __init__(self, cache_path: Path, model: str = MODEL, batch_size: int = 24) -> None:
        self.model_name = model
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.cache: dict[str, Generation] = {}
        self.generated_seconds = 0.0
        self._loaded: tuple[Any, Any] | None = None
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    key = row.pop("key")
                    self.cache[key] = Generation(**row)

    def _load(self) -> tuple[Any, Any]:
        if self._loaded is None:
            mlx_lm: Any = importlib.import_module("mlx_lm")
            model, tokenizer = mlx_lm.load(self.model_name)[:2]
            self._loaded = (model, tokenizer)
        return self._loaded

    def generate(
        self, requests: Sequence[Request], max_tokens: int, early_stop: Sequence[str] = ()
    ) -> list[Generation]:
        keys = [cache_key(self.model_name, r, max_tokens, early_stop) for r in requests]
        todo = list(dict.fromkeys(k for k in keys if k not in self.cache))
        by_key = dict(zip(keys, requests, strict=True))
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i : i + self.batch_size]
            gens = self._run([by_key[k] for k in chunk], max_tokens, early_stop)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                for k, g in zip(chunk, gens, strict=True):
                    self.cache[k] = g
                    f.write(json.dumps({"key": k, **asdict(g)}, ensure_ascii=False) + "\n")
        return [self.cache[k] for k in keys]

    def _run(
        self, requests: list[Request], max_tokens: int, early_stop: Sequence[str]
    ) -> list[Generation]:
        mg: Any = importlib.import_module("mlx_lm.generate")
        model, tokenizer = self._load()
        eos = set(tokenizer.eos_token_ids)
        prompts = []
        for r in requests:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": r.user}], add_generation_prompt=True, tokenize=False
            )
            prompts.append(tokenizer.encode(text + r.prefix, add_special_tokens=False))
        # 停止文字列はトークン列で早めに打ち切るだけで、正確な切り取りは呼び出し側が文字列で行う
        stops = mg.make_stop_sequences(tokenizer, list(early_stop))
        start = time.perf_counter()
        gen = mg.BatchGenerator(
            model,
            stop_tokens=[[t] for t in eos],
            completion_batch_size=self.batch_size,
        )
        uids = gen.insert(
            prompts, [max_tokens] * len(prompts), stop_sequences=[stops] * len(prompts)
        )
        tokens: dict[int, list[int]] = {u: [] for u in uids}
        while responses := gen.next_generated():
            for res in responses:
                if res.token not in eos:
                    tokens[res.uid].append(res.token)
        gen.close()
        elapsed = time.perf_counter() - start
        self.generated_seconds += elapsed
        share = elapsed / len(requests)
        print(
            f"  generated {len(requests)} in {elapsed:.1f}s (total {self.generated_seconds:.0f}s)",
            flush=True,
        )
        return [Generation(tokenizer.decode(tokens[u]), len(tokens[u]), share) for u in uids]
