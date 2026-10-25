from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llm import LLM, Generation, Request
from prompts import closed_user, rag_user, react_prefix, react_user
from questions import Question
from retrieval import article_numbers, containment
from tools import (
    LABEL_RE,
    Action,
    Environment,
    first_finish,
    labeled_segments,
    parse_action,
    stop_strings,
    truncate_at_stop,
)

MALFORMED_OBSERVATION = (
    "行動を解析できなかった。search[クエリ] / lookup[語] / finish[回答] のどれかで書く"
)
MISMATCH_THRESHOLD = 0.5
EARLY_STOP = ("\n観測",)


@dataclass(frozen=True)
class Fabrication:
    action: Action | None
    written: str
    actual: str
    overlap: float

    @property
    def mismatch(self) -> bool:
        if not article_numbers(self.written) <= article_numbers(self.actual):
            return True
        return self.overlap < MISMATCH_THRESHOLD


@dataclass
class Outcome:
    question: Question
    answer: str = ""
    env: Environment | None = None
    calls: int = 0
    seconds: float = 0.0
    tokens: int = 0
    forced: bool = False
    malformed: int = 0
    fabrications: list[Fabrication] = field(default_factory=list)
    transcript: str = ""

    @property
    def retrieved(self) -> set[int]:
        return self.env.retrieved if self.env else set()

    @property
    def searches(self) -> int:
        return self.env.searches if self.env else 0

    @property
    def lookups(self) -> int:
        return self.env.lookups if self.env else 0

    def record(self, gen: Generation) -> None:
        self.calls += 1
        self.seconds += gen.seconds
        self.tokens += gen.tokens


def find_fabrications(text: str, shadow: Environment) -> list[Fabrication]:
    out = []
    last: Action | None = None
    for label, seg in labeled_segments(text):
        if label == "行動":
            last = parse_action(seg)
        elif label == "観測":
            written = LABEL_RE.sub("", seg, count=1).strip()
            actual = shadow.execute(last) if last and last.tool != "finish" else ""
            out.append(Fabrication(last, written, actual, containment(written, actual)))
            last = None
    return out


@dataclass
class Episode:
    outcome: Outcome
    env: Environment
    max_steps: int
    use_stop: bool
    done: bool = False
    steps: int = 0

    def request(self) -> Request:
        return Request(
            react_user(self.outcome.question.text),
            react_prefix(self.outcome.transcript, self.steps + 1),
        )

    def advance(self, gen: Generation) -> None:
        self.steps += 1
        n = self.steps
        o = self.outcome
        o.record(gen)
        if self.use_stop:
            text, _ = truncate_at_stop(gen.text, stop_strings(n))
        else:
            text = gen.text
            o.fabrications.extend(find_fabrications(text, self.env.clone()))
        action = parse_action(text)
        finish = first_finish(text) if not self.use_stop else None
        observation = None
        if finish is not None:
            o.answer, self.done = finish.arg, True
        elif action is None:
            o.malformed += 1
            observation = MALFORMED_OBSERVATION
        elif action.tool == "finish":
            o.answer, self.done = action.arg, True
        else:
            observation = self.env.execute(action)
        o.transcript += f"思考 {n}:{text.rstrip()}\n"
        if observation is not None:
            o.transcript += f"観測 {n}: {observation}\n"
        if not self.done and n >= self.max_steps:
            o.forced, self.done = True, True


def run_react(
    questions: Sequence[Question],
    make_env: Callable[[], Environment],
    llm: LLM,
    max_steps: int,
    use_stop: bool = True,
    max_tokens: int = 160,
) -> list[Outcome]:
    episodes = []
    for q in questions:
        env = make_env()
        episodes.append(Episode(Outcome(q, env=env), env, max_steps, use_stop))
    while active := [e for e in episodes if not e.done]:
        gens = llm.generate(
            [e.request() for e in active], max_tokens, EARLY_STOP if use_stop else ()
        )
        for e, g in zip(active, gens, strict=True):
            e.advance(g)
    return [e.outcome for e in episodes]


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def run_rag(
    questions: Sequence[Question],
    make_env: Callable[[], Environment],
    llm: LLM,
    max_tokens: int = 96,
) -> list[Outcome]:
    outcomes = []
    requests = []
    for q in questions:
        env = make_env()
        env.search(q.text)
        outcomes.append(Outcome(q, env=env))
        requests.append(Request(rag_user(q.text, env.opened)))
    for o, g in zip(outcomes, llm.generate(requests, max_tokens), strict=True):
        o.record(g)
        o.transcript = g.text
        o.answer = first_line(g.text)
    return outcomes


def run_closed(questions: Sequence[Question], llm: LLM, max_tokens: int = 96) -> list[Outcome]:
    outcomes = [Outcome(q) for q in questions]
    gens = llm.generate([Request(closed_user(q.text)) for q in questions], max_tokens)
    for o, g in zip(outcomes, gens, strict=True):
        o.record(g)
        o.transcript = g.text
        o.answer = first_line(g.text)
    return outcomes
