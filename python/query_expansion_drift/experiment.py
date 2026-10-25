from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

import dense
from bm25 import BM25, expand, tokenize
from data import Dataset, load_scifact, sample_ids
from generate import MODEL, T1_SAMPLES, Generation, generate_all, key
from metrics import ndcg_at_k, overlap_at_k, rank, recall_at_k
from prompts import FORMS, pick_examples

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FORM_LABEL = {"query2doc": "Query2doc", "hyde": "HyDE", "properties": "性質の列挙"}
VARIANTS: list[tuple[float, int]] = [(0.0, 0)] + [(1.0, s) for s in range(T1_SAMPLES)]
REPEATS = (1, 3, 5)
ALPHAS = (0.8, 0.6, 0.4, 0.2, 0.0)
BASE = {"bm25": "元のクエリだけ", "dense": "α=1（元のクエリだけ）"}
MEAN = "元のクエリと T=1の3本の平均"
BM25_SETTINGS = ["生成文だけ", *(f"元のクエリ×{n} + 生成文" for n in REPEATS)]
DENSE_SETTINGS = [
    *(f"α={a:g}" + ("（生成文だけ）" if a == 0 else "") for a in ALPHAS),
    "連結した文字列",
]


@dataclass(frozen=True)
class Result:
    ndcg: np.ndarray
    recall: np.ndarray
    ranked: list[list[str]]


RunKey = tuple[str, str, str, float, int]


def evaluate(
    score_matrix: np.ndarray, qids: list[str], doc_ids: list[str], qrels: dict[str, dict[str, int]]
) -> Result:
    top = rank(score_matrix, 100)
    ranked = [[doc_ids[i] for i in row] for row in top]
    ndcg = np.array([ndcg_at_k(r, qrels[q], 10) for r, q in zip(ranked, qids, strict=True)])
    recall = np.array([recall_at_k(r, qrels[q], 100) for r, q in zip(ranked, qids, strict=True)])
    return Result(ndcg, recall, ranked)


def corpus_embeddings(model: Any, texts: list[str], path: Path) -> np.ndarray:
    if path.exists():
        cached = np.load(path)
        if cached.shape[0] == len(texts):
            return np.asarray(cached)
    emb = np.asarray(model.encode(texts, batch_size=64, normalize_embeddings=True))
    np.save(path, emb)
    return emb


def run_all(
    ds: Dataset, qids: list[str], gens: dict[str, Generation], cache_dir: Path
) -> tuple[dict[RunKey, Result], dict[str, float]]:
    from sentence_transformers import SentenceTransformer

    timings: dict[str, float] = {}
    doc_ids = [d.doc_id for d in ds.docs]
    doc_texts = [f"{d.title} {d.text}" for d in ds.docs]
    queries = [ds.queries[q] for q in qids]
    qrels = ds.test_qrels

    def ev(m: np.ndarray) -> Result:
        return evaluate(m, qids, doc_ids, qrels)

    def text(form: str, temp: float, sample: int) -> list[str]:
        return [gens[key(q, form, temp, sample)].text for q in qids]

    runs: dict[RunKey, Result] = {}
    start = time.perf_counter()
    index = BM25([tokenize(t) for t in doc_texts])
    q_toks = [tokenize(q) for q in queries]
    runs[("bm25", BASE["bm25"], "", 0.0, 0)] = ev(index.scores(q_toks))
    for form, (temp, sample) in itertools.product(FORMS, VARIANTS):
        g_toks = [tokenize(t) for t in text(form, temp, sample)]
        runs[("bm25", BM25_SETTINGS[0], form, temp, sample)] = ev(index.scores(g_toks))
        for n, name in zip(REPEATS, BM25_SETTINGS[1:], strict=True):
            mixed = [expand(q, g, n) for q, g in zip(q_toks, g_toks, strict=True)]
            runs[("bm25", name, form, temp, sample)] = ev(index.scores(mixed))
    timings["bm25"] = time.perf_counter() - start

    start = time.perf_counter()
    model = SentenceTransformer(EMBED_MODEL)
    doc_emb = corpus_embeddings(model, doc_texts, cache_dir / "corpus_minilm.npy")
    timings["corpus_encode"] = time.perf_counter() - start

    start = time.perf_counter()

    def enc(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, batch_size=64, normalize_embeddings=True))

    e_q = enc(queries)
    runs[("dense", BASE["dense"], "", 0.0, 0)] = ev(dense.scores(doc_emb, e_q))
    e_gen: dict[tuple[str, float, int], np.ndarray] = {}
    for form, (temp, sample) in itertools.product(FORMS, VARIANTS):
        gen_texts = text(form, temp, sample)
        e_g = enc(gen_texts)
        e_gen[(form, temp, sample)] = e_g
        for alpha, name in zip(ALPHAS, DENSE_SETTINGS[:-1], strict=True):
            vec = dense.interpolate(e_q, e_g, alpha)
            runs[("dense", name, form, temp, sample)] = ev(dense.scores(doc_emb, vec))
        concat = enc([f"{q} {g}" for q, g in zip(queries, gen_texts, strict=True)])
        runs[("dense", DENSE_SETTINGS[-1], form, temp, sample)] = ev(dense.scores(doc_emb, concat))
    for form in FORMS:
        samples = [e_gen[(form, 1.0, s)] for s in range(T1_SAMPLES)]
        vec = dense.mean_with_query(e_q, samples)
        runs[("dense", MEAN, form, 1.0, -1)] = ev(dense.scores(doc_emb, vec))
    timings["dense"] = time.perf_counter() - start
    return runs, timings


def fmt(x: float) -> str:
    return f"{x:.3f}"


def table(header: list[str], rows: list[list[str]], text_cols: int = 1) -> str:
    sep = ["---" if i < text_cols else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def t0_key(retriever: str, name: str, form: str) -> RunKey:
    if name == MEAN:
        return (retriever, name, form, 1.0, -1)
    if name == BASE[retriever]:
        return (retriever, name, "", 0.0, 0)
    return (retriever, name, form, 0.0, 0)


def settings_for(retriever: str) -> list[str]:
    if retriever == "bm25":
        return [BASE["bm25"], *BM25_SETTINGS]
    return [BASE["dense"], *DENSE_SETTINGS, MEAN]


def quality_table(runs: dict[RunKey, Result], retriever: str) -> str:
    header = ["設定", *(f"{FORM_LABEL[f]} nDCG@10" for f in FORMS)]
    header += [f"{FORM_LABEL[f]} Recall@100" for f in FORMS]
    rows = []
    for name in settings_for(retriever):
        res = [runs[t0_key(retriever, name, f)] for f in FORMS]
        rows.append(
            [name, *(fmt(r.ndcg.mean()) for r in res), *(fmt(r.recall.mean()) for r in res)]
        )
    return table(header, rows)


def drift_table(runs: dict[RunKey, Result], retriever: str) -> str:
    base = runs[(retriever, BASE[retriever], "", 0.0, 0)].ndcg
    header = ["設定"]
    for f in FORMS:
        header += [f"{FORM_LABEL[f]} 悪化 / 改善", f"{FORM_LABEL[f]} 平均の差 [95%区間]"]
    rows = []
    for name in settings_for(retriever)[1:]:
        row = [name]
        for f in FORMS:
            delta = runs[t0_key(retriever, name, f)].ndcg - base
            worse = int((delta < -1e-9).sum())
            better = int((delta > 1e-9).sum())
            half = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
            row += [
                f"{worse} / {better}",
                f"{delta.mean():+.3f} [{delta.mean() - half:+.3f}, {delta.mean() + half:+.3f}]",
            ]
        rows.append(row)
    return table(header, rows)


def temperature_rows(runs: dict[RunKey, Result], retriever: str) -> list[list[str]]:
    names = BM25_SETTINGS if retriever == "bm25" else DENSE_SETTINGS
    rows = []
    for f in FORMS:
        for name in names:
            t0 = runs[(retriever, name, f, 0.0, 0)]
            t1 = [runs[(retriever, name, f, 1.0, s)] for s in range(T1_SAMPLES)]
            means = [r.ndcg.mean() for r in t1]
            pairs = list(itertools.combinations(t1, 2))
            overlap = np.mean(
                [
                    np.mean(
                        [overlap_at_k(a, b, 10) for a, b in zip(x.ranked, y.ranked, strict=True)]
                    )
                    for x, y in pairs
                ]
            )
            rows.append(
                [
                    FORM_LABEL[f],
                    name,
                    fmt(t0.ndcg.mean()),
                    fmt(float(np.mean(means))),
                    f"{fmt(min(means))}〜{fmt(max(means))}",
                    fmt(max(means) - min(means)),
                    fmt(float(overlap)),
                ]
            )
    return rows


def temperature_table(runs: dict[RunKey, Result], retriever: str) -> str:
    header = ["形式", "設定", "T=0", "T=1 平均", "T=1 最小〜最大", "振れ幅", "上位10件の重なり"]
    return table(header, temperature_rows(runs, retriever), text_cols=2)


def _pos(ranked: list[str], doc: str) -> int | None:
    return ranked.index(doc) + 1 if doc in ranked else None


def worst_examples(
    runs: dict[RunKey, Result],
    ds: Dataset,
    qids: list[str],
    gens: dict[str, Generation],
    run_key: RunKey,
    n: int = 3,
) -> list[dict[str, Any]]:
    retriever, _, form, temp, sample = run_key
    base = runs[(retriever, BASE[retriever], "", 0.0, 0)]
    res = runs[run_key]
    titles = {d.doc_id: d.title for d in ds.docs}
    delta = res.ndcg - base.ndcg
    out = []
    for i in np.argsort(delta, kind="stable")[:n]:
        if delta[i] >= 0:
            break
        q = qids[i]
        rel = ds.test_qrels[q]
        lost = [d for d in base.ranked[i][:10] if d in rel and d not in res.ranked[i][:10]]
        generated = gens[key(q, form, temp, max(sample, 0))].text
        out.append(
            {
                "query": ds.queries[q],
                "generated": generated[:160],
                "ndcg": [round(float(base.ndcg[i]), 3), round(float(res.ndcg[i]), 3)],
                "lost": [
                    {"title": titles[d], "rank": [_pos(base.ranked[i], d), _pos(res.ranked[i], d)]}
                    for d in lost
                ],
            }
        )
    return out


def env() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.mac_ver()[0]} {platform.machine()}",
        **{
            p: version(p)
            for p in ("mlx", "mlx-lm", "sentence-transformers", "torch", "numpy", "scipy")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    args = parser.parse_args()

    total = time.perf_counter()
    ds = load_scifact(args.cache_dir / "scifact.zip")
    qids = sample_ids(list(ds.test_qrels), args.n_queries, args.seed)
    examples = pick_examples(ds, 4, args.seed)
    print(f"docs={len(ds.docs)} test_queries={len(ds.test_qrels)} used={len(qids)}")
    gens, gen_stats = generate_all(
        qids, ds.queries, examples, args.cache_dir / "generations.jsonl", seed=args.seed
    )
    runs, timings = run_all(ds, qids, gens, args.cache_dir)

    used = [gens[key(q, f, t, s)] for q in qids for f in FORMS for t, s in VARIANTS]
    avg_tokens = {f: float(np.mean([g.n_tokens for g in used if g.form == f])) for f in FORMS}
    sections: list[tuple[str, str]] = [
        ("BM25 T=0", quality_table(runs, "bm25")),
        ("密ベクトル T=0", quality_table(runs, "dense")),
        ("BM25 ドリフト T=0", drift_table(runs, "bm25")),
        ("密ベクトル ドリフト T=0", drift_table(runs, "dense")),
        ("BM25 temperature", temperature_table(runs, "bm25")),
        ("密ベクトル temperature", temperature_table(runs, "dense")),
    ]
    example_keys: list[RunKey] = [
        ("bm25", "元のクエリ×5 + 生成文", "query2doc", 0.0, 0),
        ("bm25", "生成文だけ", "properties", 0.0, 0),
        ("dense", "α=0（生成文だけ）", "hyde", 0.0, 0),
        ("dense", MEAN, "hyde", 1.0, -1),
    ]
    examples_out = {
        " / ".join(map(str, k)): worst_examples(runs, ds, qids, gens, k) for k in example_keys
    }
    for title, body in sections:
        print(f"\n### {title}\n\n{body}")
    print("\n### 悪化した例")
    print(json.dumps(examples_out, ensure_ascii=False, indent=1))
    summary = {
        "env": env(),
        "llm": MODEL,
        "embedding": EMBED_MODEL,
        "n_queries": len(qids),
        "generation": gen_stats,
        "avg_generated_tokens": avg_tokens,
        "timings": timings,
        "total_seconds": time.perf_counter() - total,
    }
    print("\n### 実行情報")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    (args.cache_dir / "results.md").write_text(
        "\n\n".join(f"### {t}\n\n{b}" for t, b in sections) + "\n"
    )
    (args.cache_dir / "summary.json").write_text(
        json.dumps({**summary, "examples": examples_out}, ensure_ascii=False, indent=1)
    )


if __name__ == "__main__":
    main()
