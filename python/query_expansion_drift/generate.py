from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from prompts import FORMS, MAX_TOKENS, Example, build_prompt, clean

MODEL = "mlx-community/gemma-3-4b-it-4bit"
T1_SAMPLES = 3


@dataclass(frozen=True)
class Generation:
    query_id: str
    form: str
    temperature: float
    sample: int
    text: str
    n_tokens: int


def key(query_id: str, form: str, temperature: float, sample: int) -> str:
    return f"{query_id}|{form}|{temperature}|{sample}"


def load_cache(path: Path) -> dict[str, Generation]:
    if not path.exists():
        return {}
    out: dict[str, Generation] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            g = Generation(**json.loads(line))
            out[key(g.query_id, g.form, g.temperature, g.sample)] = g
    return out


def jobs_for(query_ids: list[str]) -> list[tuple[str, str, float, int]]:
    jobs = []
    for qid in query_ids:
        for form in FORMS:
            jobs.append((qid, form, 0.0, 0))
            jobs.extend((qid, form, 1.0, s) for s in range(T1_SAMPLES))
    return jobs


def _encode(tokenizer: Any, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    return ids


def generate_all(
    query_ids: list[str],
    queries: dict[str, str],
    examples: list[Example],
    cache_path: Path,
    chunk: int = 8,
    seed: int = 0,
    batch_size: int = 16,
) -> tuple[dict[str, Generation], dict[str, float]]:
    import mlx.core as mx
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    cache = load_cache(cache_path)
    todo = [j for j in jobs_for(query_ids) if key(*j) not in cache]
    stats = {"seconds": 0.0, "generated": 0.0, "tokens": 0.0}
    if not todo:
        return cache, stats

    loaded = load(MODEL)
    model, tokenizer = loaded[0], loaded[1]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    step = chunk * len(FORMS) * (1 + T1_SAMPLES)
    start = time.perf_counter()
    for i in range(0, len(todo), step):
        part = todo[i : i + step]
        new: list[Generation] = []
        for temperature in (0.0, 1.0):
            group = [j for j in part if j[2] == temperature]
            if not group:
                continue
            prompts = [
                _encode(tokenizer, build_prompt(j[1], queries[j[0]], examples)) for j in group
            ]
            # GPU のメモリをほかの MLX の処理と分け合うので、足りなければ幅を半分にして流し直す
            while True:
                mx.random.seed(seed + i)
                try:
                    res = batch_generate(
                        model,
                        tokenizer,
                        prompts,
                        max_tokens=[MAX_TOKENS[j[1]] for j in group],
                        sampler=make_sampler(temp=temperature),
                        completion_batch_size=batch_size,
                        return_token_ids=True,
                    )
                    break
                except RuntimeError as e:
                    if batch_size <= 1 or "Memory" not in str(e):
                        raise
                    mx.clear_cache()
                    batch_size //= 2
                    print(f"  out of memory, retry with batch_size={batch_size}", flush=True)
            token_ids = res.token_ids or [[] for _ in group]
            for j, text, ids in zip(group, res.texts, token_ids, strict=True):
                new.append(Generation(j[0], j[1], j[2], j[3], clean(text), len(ids)))
        with cache_path.open("a") as f:
            for g in new:
                f.write(json.dumps(asdict(g), ensure_ascii=False) + "\n")
                cache[key(g.query_id, g.form, g.temperature, g.sample)] = g
        stats["generated"] += len(new)
        stats["tokens"] += sum(g.n_tokens for g in new)
        elapsed = time.perf_counter() - start
        print(f"  generated {int(stats['generated'])}/{len(todo)} in {elapsed:.0f}s", flush=True)
    stats["seconds"] = time.perf_counter() - start
    return cache, stats
