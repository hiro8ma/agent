from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from llm import MODEL, MLXModel
from metrics import Summary, answer_score, evidence_recall, mean, summarize
from questions import KINDS, QUESTIONS, Question
from react import Outcome, run_closed, run_rag, run_react
from retrieval import BM25, Article, load_articles, substring_search
from tools import Environment

LABELS = {
    "closed": "(参考) 検索なし",
    "rag": "(a) 1回の検索",
    "react2": "(b) ReAct 上限2回",
    "react5": "(c) ReAct 上限5回",
    "nostop": "(d) 停止文字列なし 上限5回",
}


def select(questions: Sequence[Question], per_kind: int | None) -> list[Question]:
    if per_kind is None:
        return list(questions)
    out = []
    for kind in KINDS:
        out.extend([q for q in questions if q.kind == kind][:per_kind])
    return out


def run_setting(
    name: str,
    questions: list[Question],
    make_env: Callable[[], Environment],
    llm: MLXModel,
    nostop_tokens: int,
) -> list[Outcome]:
    if name == "closed":
        return run_closed(questions, llm)
    if name == "rag":
        return run_rag(questions, make_env, llm)
    if name == "react2":
        return run_react(questions, make_env, llm, max_steps=2)
    if name == "react5":
        return run_react(questions, make_env, llm, max_steps=5)
    if name == "nostop":
        return run_react(
            questions, make_env, llm, max_steps=5, use_stop=False, max_tokens=nostop_tokens
        )
    raise ValueError(name)


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def overall_table(results: dict[str, list[Outcome]]) -> str:
    rows = [
        "| 設定 | 正答率 | 根拠の再現率 | 検索の回数 | lookup の回数 | LLM の呼び出し | 生成トークン | 秒/問 | 強制終了 | 形式崩れ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, outcomes in results.items():
        s = summarize(outcomes)
        ev = "-" if name == "closed" else pct(s.evidence)
        rows.append(
            f"| {LABELS[name]} | {pct(s.accuracy)} | {ev} | {s.searches:.1f} | {s.lookups:.1f} "
            f"| {s.calls:.1f} | {s.tokens:.0f} | {s.seconds:.1f} | {s.forced} "
            f"| {s.malformed_questions}問（{s.malformed_steps}回） |"
        )
    return "\n".join(rows)


def kind_table(
    results: dict[str, list[Outcome]], metric: Callable[[Outcome], float], skip: set[str]
) -> str:
    names = [n for n in results if n not in skip]
    rows = [
        "| 種類 | 問い数 | " + " | ".join(LABELS[n] for n in names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for kind, label in KINDS.items():
        cells = []
        count = 0
        for n in names:
            picked = [o for o in results[n] if o.question.kind == kind]
            count = len(picked)
            cells.append(pct(mean([metric(o) for o in picked])))
        if count:
            rows.append(f"| {label} | {count} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def fabrication_table(outcomes: list[Outcome]) -> str:
    s: Summary = summarize(outcomes)
    return "\n".join(
        [
            "| 問い数 | 観測を自分で書いた問い | 書いた観測の数 | 実際の結果と食い違った観測 | 本物の検索を1回もせずに回答した問い |",
            "|---|---|---|---|---|",
            f"| {s.n} | {s.fabricated_questions} | {s.fabrications} | {s.mismatches} | {s.answered_without_search} |",
        ]
    )


def retrieval_table(
    questions: list[Question],
    articles: list[Article],
    bm25: BM25,
    react_outcomes: list[Outcome] | None,
) -> str:
    def recall(q: Question, queries: list[str], search: Callable[[str], list[Article]]) -> float:
        got = {a.number for query in queries for a in search(query)}
        return evidence_recall(got, q)

    def bm(query: str) -> list[Article]:
        return bm25.search(query)

    def sub(query: str) -> list[Article]:
        return substring_search(articles, query)

    queries = {o.question.qid: o.env.queries for o in react_outcomes or [] if o.env}
    header = "| 種類 | 問い数 | BM25 問いそのまま | 部分一致 問いそのまま |"
    sep = "|---|---|---|---|"
    if queries:
        header += " BM25 ReAct のクエリ | 部分一致 ReAct のクエリ |"
        sep += "---|---|"
    rows = [header, sep]
    for kind, label in [*KINDS.items(), ("all", "全体")]:
        picked = [q for q in questions if kind in ("all", q.kind)]
        if not picked:
            continue
        cells = [
            pct(mean([recall(q, [q.text], bm) for q in picked])),
            pct(mean([recall(q, [q.text], sub) for q in picked])),
        ]
        if queries:
            cells.append(pct(mean([recall(q, queries.get(q.qid, []), bm) for q in picked])))
            cells.append(pct(mean([recall(q, queries.get(q.qid, []), sub) for q in picked])))
        rows.append(f"| {label} | {len(picked)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def dump(path: Path, results: dict[str, list[Outcome]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for name, outcomes in results.items():
            for o in outcomes:
                row = {
                    "setting": name,
                    "qid": o.question.qid,
                    "kind": o.question.kind,
                    "question": o.question.text,
                    "answer": o.answer,
                    "score": answer_score(o.answer, o.question),
                    "evidence": evidence_recall(o.retrieved, o.question),
                    "retrieved": sorted(o.retrieved),
                    "queries": o.env.queries if o.env else [],
                    "forced": o.forced,
                    "malformed": o.malformed,
                    "fabrications": [asdict(fab) for fab in o.fabrications],
                    "transcript": o.transcript,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-kind", type=int, default=None)
    parser.add_argument("--settings", nargs="+", default=list(LABELS), choices=list(LABELS))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--nostop-tokens", type=int, default=512)
    args = parser.parse_args()

    articles = load_articles()
    bm25 = BM25(articles)

    def make_env() -> Environment:
        return Environment(bm25.search, articles)

    questions = select(QUESTIONS, args.per_kind)
    llm = MLXModel(args.cache_dir / "generations.jsonl", batch_size=args.batch_size)
    results: dict[str, list[Outcome]] = {}
    wall: dict[str, float] = {}
    for name in args.settings:
        print(f"[{LABELS[name]}] {len(questions)} questions", flush=True)
        start = time.perf_counter()
        results[name] = run_setting(name, questions, make_env, llm, args.nostop_tokens)
        wall[name] = time.perf_counter() - start

    dump(args.cache_dir / "outcomes.jsonl", results)
    print(f"\nmodel {MODEL}, questions {len(questions)}")
    print("wall seconds " + ", ".join(f"{n} {s:.1f}" for n, s in wall.items()))
    print(f"generation seconds this run {llm.generated_seconds:.0f}\n")
    print(overall_table(results), "\n")
    print(kind_table(results, lambda o: answer_score(o.answer, o.question), set()), "\n")
    print(kind_table(results, lambda o: evidence_recall(o.retrieved, o.question), {"closed"}), "\n")
    if "nostop" in results:
        print(fabrication_table(results["nostop"]), "\n")
    print(retrieval_table(questions, articles, bm25, results.get("react5")))


if __name__ == "__main__":
    main()
