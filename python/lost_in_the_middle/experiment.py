from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from data import Example, download, load_examples, sample_examples
from generate import MAX_TOKENS, CachedGenerator, model_slug
from metrics import first_line, is_correct
from prompts import (
    QuestionPlacement,
    build_closedbook_prompt,
    build_qa_prompt,
    place_gold,
    reorder_ends,
    reordered_position,
)
from ranks import assign_ranks, uniform_weights, zipf_weights

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "mlx-community/gemma-3-4b-it-4bit"
POSITIONS = {5: (1, 3, 5), 10: (1, 5, 10), 20: (1, 5, 10, 15, 20)}
QUERY_POSITIONS = (1, 10, 20)
MIDDLE = {5: 3, 10: 5, 20: 10}
REORDER_K = 20
EXPERIMENTS = ("baseline", "position", "reorder", "k", "query")
DISTRIBUTIONS: dict[str, Callable[[int], list[float]]] = {
    "上位に偏る（1/r に比例）": zipf_weights,
    "一様": uniform_weights,
}
RANK_BUCKETS = ((1, 1), (2, 5), (6, 10), (11, 20))
PROGRESS_EVERY = 25


@dataclass(frozen=True)
class ReorderTrial:
    distribution: str
    rank: int
    ranked_correct: bool
    reordered_correct: bool


class Judge:
    def __init__(self, generate: CachedGenerator) -> None:
        self.generate = generate
        self.calls = 0
        self.started = time.perf_counter()

    def __call__(self, prompt: str, example: Example) -> bool:
        output = self.generate(prompt)
        self.calls += 1
        if self.calls % PROGRESS_EVERY == 0:
            elapsed = time.perf_counter() - self.started
            print(
                f"{self.calls} prompts, {self.generate.new_generations} new, {elapsed:.0f}s",
                file=sys.stderr,
                flush=True,
            )
        return is_correct(output, example.answers)


def percent(flags: Sequence[bool]) -> float:
    return 100 * sum(flags) / len(flags)


def run_positions(
    judge: Judge,
    examples: Sequence[Example],
    k: int,
    positions: Sequence[int],
    placement: QuestionPlacement,
) -> dict[int, list[bool]]:
    results = {}
    for p in positions:
        flags = []
        for ex in examples:
            docs = place_gold(ex.gold, ex.distractors, k, p)
            flags.append(judge(build_qa_prompt(ex.question, docs, placement), ex))
        results[p] = flags
    return results


def run_reorder(judge: Judge, examples: Sequence[Example], seed: int) -> list[ReorderTrial]:
    trials = []
    for offset, (name, weights) in enumerate(DISTRIBUTIONS.items()):
        ranks = assign_ranks(len(examples), weights(REORDER_K), seed + offset)
        for ex, rank in zip(examples, ranks, strict=True):
            ranked = place_gold(ex.gold, ex.distractors, REORDER_K, rank)
            trials.append(
                ReorderTrial(
                    distribution=name,
                    rank=rank,
                    ranked_correct=judge(build_qa_prompt(ex.question, ranked, "after"), ex),
                    reordered_correct=judge(
                        build_qa_prompt(ex.question, reorder_ends(ranked), "after"), ex
                    ),
                )
            )
    return trials


def row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def fmt(value: float) -> str:
    return f"{value:.1f}%"


def print_positions(results: dict[str, dict[int, list[bool]]], n: int) -> None:
    print(f"### 正解文書の位置（K=20、{n}問）\n")
    positions = POSITIONS[20]
    print(row(["質問の位置", *[f"位置{p}" for p in positions]]))
    print("|---|" + "---:|" * len(positions))
    labels = {"after": "文書の後だけ", "both": "文書の前後"}
    for placement, by_position in results.items():
        cells = [fmt(percent(by_position[p])) if p in by_position else "-" for p in positions]
        print(row([labels[placement], *cells]))
    print()
    if "after" in results:
        hits = [sum(flags) for flags in zip(*results["after"].values(), strict=True)]
        print(
            f"文書の後だけの5か所で、すべて正解{hits.count(len(positions))}問、"
            f"すべて不正解{hits.count(0)}問、位置で正誤が変わった"
            f"{n - hits.count(len(positions)) - hits.count(0)}問\n"
        )


def print_k(results: dict[int, dict[int, list[bool]]], n: int) -> None:
    print(f"### 文書数 K（質問は文書の後だけ、{n}問）\n")
    print(row(["K", "先頭", "中央", "末尾", "3か所の平均", "中央の位置"]))
    print("|---:|---:|---:|---:|---:|---:|")
    for k, by_position in sorted(results.items()):
        cells = [percent(by_position[p]) for p in (1, MIDDLE[k], k)]
        print(row([k, *[fmt(c) for c in cells], fmt(sum(cells) / 3), MIDDLE[k]]))
    print()


def print_reorder(trials: Sequence[ReorderTrial]) -> None:
    print(f"### 並べ替え（K={REORDER_K}、質問は文書の後だけ）\n")
    print(
        row(
            [
                "r の分布",
                "問数",
                "(a) 関連度順",
                "(b) 両端に寄せる",
                "b - a",
                "b だけ正解",
                "a だけ正解",
            ]
        )
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name in DISTRIBUTIONS:
        ts = [t for t in trials if t.distribution == name]
        a = percent([t.ranked_correct for t in ts])
        b = percent([t.reordered_correct for t in ts])
        b_only = sum(t.reordered_correct and not t.ranked_correct for t in ts)
        a_only = sum(t.ranked_correct and not t.reordered_correct for t in ts)
        print(row([name, len(ts), fmt(a), fmt(b), f"{b - a:+.1f}", b_only, a_only]))
    print()
    print("#### r ごとの内訳（2つの分布をまとめる）\n")
    print(row(["r", "件数", "(a) の正解文書の位置", "(a)", "(b) の正解文書の位置", "(b)"]))
    print("|---|---:|---|---:|---|---:|")
    for low, high in RANK_BUCKETS:
        ts = [t for t in trials if low <= t.rank <= high]
        if not ts:
            continue
        label = str(low) if low == high else f"{low}〜{high}"
        b_positions = ", ".join(str(reordered_position(r, REORDER_K)) for r in range(low, high + 1))
        print(
            row(
                [
                    label,
                    len(ts),
                    label,
                    fmt(percent([t.ranked_correct for t in ts])),
                    b_positions,
                    fmt(percent([t.reordered_correct for t in ts])),
                ]
            )
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "qa_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    examples = sample_examples(load_examples(download(args.data_dir)), args.questions, args.seed)
    generate = CachedGenerator(args.model, args.output_dir / "cache")
    judge = Judge(generate)
    results: dict[str, object] = {
        "model": args.model,
        "questions": args.questions,
        "seed": args.seed,
    }

    baseline: dict[str, float] = {}
    positions: dict[str, dict[int, list[bool]]] = {}
    by_k: dict[int, dict[int, list[bool]]] = {}
    trials: list[ReorderTrial] = []
    for name in args.experiments:
        if name == "baseline":
            baseline["closedbook"] = percent(
                [judge(build_closedbook_prompt(ex.question), ex) for ex in examples]
            )
            baseline["oracle"] = percent(
                [judge(build_qa_prompt(ex.question, [ex.gold], "after"), ex) for ex in examples]
            )
        elif name == "position":
            positions["after"] = run_positions(judge, examples, 20, POSITIONS[20], "after")
        elif name == "query":
            positions["both"] = run_positions(judge, examples, 20, QUERY_POSITIONS, "both")
        elif name == "k":
            for k in (5, 10, 20):
                by_k[k] = run_positions(judge, examples, k, POSITIONS[k], "after")
        elif name == "reorder":
            trials = run_reorder(judge, examples, args.seed)

    print(f"## {args.model}\n")
    print(f"- Python {sys.version.split()[0]}、{platform.platform()}")
    print(f"- mlx {version('mlx')}、mlx-lm {version('mlx-lm')}")
    print(f"- 質問{args.questions}問、seed {args.seed}、貪欲デコード、最大{MAX_TOKENS}トークン")
    print(
        f"- 使ったプロンプト{len(generate.used)}件、"
        f"生成時間の合計{generate.used_seconds() / 60:.1f}分"
        f"（この実行で新しく生成したのは{generate.new_generations}件）"
    )
    lead_ins = sum(first_line(generate.outputs[k]).endswith(":") for k in generate.used)
    print(f"- 最初の行が「:」で終わる前置きだった出力{lead_ins}件\n")
    if baseline:
        print(row(["条件", "正答率"]))
        print("|---|---:|")
        print(row(["文書なし（closed-book）", fmt(baseline["closedbook"])]))
        print(row(["正解文書だけ（oracle）", fmt(baseline["oracle"])]))
        print()
    if positions:
        print_positions(positions, args.questions)
    if by_k:
        print_k(by_k, args.questions)
    if trials:
        print_reorder(trials)

    results |= {
        "baseline": baseline,
        "positions": positions,
        "k": by_k,
        "reorder": [asdict(t) for t in trials],
        "prompts": len(generate.used),
        "generation_seconds": generate.used_seconds(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"results-{model_slug(args.model)}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
