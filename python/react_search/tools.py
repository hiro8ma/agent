from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from retrieval import Article, normalize

TOOLS = ("search", "lookup", "finish")
ACTION_RE = re.compile(r"行動\s*\d*\s*[:：]\s*([A-Za-z]+)\s*[\[［](.*?)[\]］]")
LABEL_RE = re.compile(r"(思考|行動|観測)\s*\d*\s*[:：]")
LOOKUP_LIMIT = 5

Searcher = Callable[[str, int], list[Article]]


@dataclass(frozen=True)
class Action:
    tool: str
    arg: str


def parse_action(text: str) -> Action | None:
    m = ACTION_RE.search(text)
    if m is None:
        return None
    tool, arg = m.group(1).lower(), m.group(2).strip()
    if tool not in TOOLS or not arg:
        return None
    return Action(tool, arg)


def stop_strings(step: int) -> tuple[str, ...]:
    return (f"観測 {step}:", "\n観測")


def truncate_at_stop(text: str, stops: Sequence[str]) -> tuple[str, bool]:
    hits = [i for i in (text.find(s) for s in stops) if i >= 0]
    if not hits:
        return text, False
    return text[: min(hits)], True


def labeled_segments(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    matches = list(LABEL_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1), text[m.start() : end]))
    return out


def first_finish(text: str) -> Action | None:
    for label, seg in labeled_segments(text):
        if label == "行動" and (a := parse_action(seg)) is not None and a.tool == "finish":
            return a
    return None


@dataclass
class Environment:
    searcher: Searcher
    articles: list[Article]
    k: int = 3
    opened: list[Article] = field(default_factory=list)
    retrieved: set[int] = field(default_factory=set)
    queries: list[str] = field(default_factory=list)
    searches: int = 0
    lookups: int = 0

    def clone(self) -> Environment:
        return Environment(
            self.searcher,
            self.articles,
            self.k,
            list(self.opened),
            set(self.retrieved),
            list(self.queries),
            self.searches,
            self.lookups,
        )

    def search(self, query: str) -> str:
        self.searches += 1
        self.queries.append(query)
        self.opened = self.searcher(query, self.k)
        self.retrieved.update(a.number for a in self.opened)
        if not self.opened:
            return "条文が見つからなかった"
        return "\n".join(a.one_line() for a in self.opened)

    def lookup(self, word: str) -> str:
        self.lookups += 1
        if not self.opened:
            return "まだ条文を開いていない。先に search を使う"
        target = normalize(word)
        lines = []
        for a in self.opened:
            for sentence in re.split(r"(?<=。)|\n", a.body):
                s = sentence.strip()
                if s and target in normalize(s):
                    lines.append(f"{a.label}　{s}")
        if not lines:
            return f"「{word}」を含む文は見つからなかった"
        return "\n".join(lines[:LOOKUP_LIMIT])

    def execute(self, action: Action) -> str:
        if action.tool == "search":
            return self.search(action.arg)
        if action.tool == "lookup":
            return self.lookup(action.arg)
        raise ValueError(f"finish は実行しない: {action}")
