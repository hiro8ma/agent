from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

MAX_TOKENS = 64


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "--")


class CachedGenerator:
    def __init__(self, model_id: str, cache_dir: Path, max_tokens: int = MAX_TOKENS) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.cache_path = cache_dir / f"{model_slug(model_id)}.jsonl"
        self.outputs: dict[str, str] = {}
        self.seconds: dict[str, float] = {}
        self.used: set[str] = set()
        if self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.outputs[row["key"]] = row["output"]
                    self.seconds[row["key"]] = row["seconds"]
        self.model: Any = None
        self.tokenizer: Any = None
        self.new_generations = 0

    def key(self, prompt: str) -> str:
        payload = json.dumps([self.model_id, self.max_tokens, prompt], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def used_seconds(self) -> float:
        return sum(self.seconds[k] for k in self.used)

    def __call__(self, prompt: str) -> str:
        key = self.key(prompt)
        self.used.add(key)
        if key in self.outputs:
            return self.outputs[key]
        from mlx_lm import load, stream_generate

        if self.model is None:
            loaded = load(self.model_id)
            self.model, self.tokenizer = loaded[0], loaded[1]
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        started = time.perf_counter()
        output = ""
        for response in stream_generate(
            self.model, self.tokenizer, chat, max_tokens=self.max_tokens
        ):
            output += response.text
            # 判定は最初の行だけを見るので、改行が出たら打ち切って時間を節約する
            if "\n" in output.strip():
                break
        seconds = time.perf_counter() - started
        self.outputs[key] = output
        self.seconds[key] = seconds
        self.new_generations += 1
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            row = {"key": key, "output": output, "seconds": round(seconds, 3)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return output
