"""The episodic reflection buffer — the only thing that "learns" between attempts.

Reflexion stores verbal self-critiques in long-term episodic memory and replays the
most recent ones into the next attempt's prompt. No gradients, no weight updates:
improvement is carried entirely by this growing text buffer (nonparametric learning).

Optionally mirrors reflections to a JSONL file (``--memory-file``) so a run can pick
up where a previous one stopped, the same way mcp/memory would persist them in SQLite.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


class ReflexionBuffer:
    """An ordered list of verbal reflections with optional JSONL persistence."""

    def __init__(self, path: str | Path | None = None, recent_window: int = 3) -> None:
        self._reflections: list[str] = []
        self._path = Path(path) if path else None
        self._recent_window = recent_window
        if self._path and self._path.exists():
            self._load()

    def add(self, reflection: str) -> None:
        text = reflection.strip()
        if not text:
            return
        self._reflections.append(text)
        if self._path:
            self._append_to_file(text)

    def recent(self) -> list[str]:
        """Return the last `recent_window` reflections (the slice injected next)."""

        if self._recent_window <= 0:
            return list(self._reflections)
        return self._reflections[-self._recent_window :]

    def render(self) -> str:
        """Format the recent reflections for injection at the head of the prompt."""

        recent = self.recent()
        if not recent:
            return ""
        lines = [f"- {r}" for r in recent]
        return "Lessons from your previous attempts (apply them):\n" + "\n".join(lines)

    def __len__(self) -> int:
        return len(self._reflections)

    def extend(self, reflections: Iterable[str]) -> None:
        for r in reflections:
            self.add(r)

    def _load(self) -> None:
        assert self._path is not None
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("reflection") if isinstance(obj, dict) else None
            if isinstance(text, str) and text.strip():
                self._reflections.append(text.strip())

    def _append_to_file(self, text: str) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"reflection": text}, ensure_ascii=False) + "\n")
