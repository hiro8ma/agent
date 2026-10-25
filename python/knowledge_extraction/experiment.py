from __future__ import annotations

import argparse
import copy
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from data import (
    ITEMS,
    Person,
    Vocab,
    augmented_bios,
    make_people,
    partially_augmented_bios,
    plain_bio,
    qa_prompt,
    reverse_prompt,
    split_people,
)
from evaluate import bio_correct, qa_correct, reverse_correct
from model import GPT, GPTConfig
from probe import majority_correct, probe_correct
from train import Example, train

ITEM_LABELS = {
    "birth_date": "生年月日",
    "birth_city": "出生地",
    "university": "大学",
    "major": "専攻",
    "employer": "勤務先",
    "work_city": "勤務地",
}


@dataclass(frozen=True)
class Config:
    n_people: int = 2_000
    n_bios: int = 5
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 64
    batch_size: int = 128
    exposures: int = 300
    pretrain_lr: float = 1e-3
    finetune_lr: float = 1e-4
    finetune_epochs: int = 40
    curve_epochs: tuple[int, ...] = (5, 10, 20, 40)
    probe_hidden: int = 256
    weight_decay: float = 0.1
    data_seed: int = 0
    model_seed: int = 0


QUICK = Config(n_people=200, n_layer=2, exposures=20, finetune_epochs=4, curve_epochs=(2, 4))


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def doc(vocab: Vocab, prompt: Sequence[str], answer: Sequence[str], answer_only: bool) -> Example:
    ids = (vocab.bos, *vocab.encode([*prompt, *answer]), vocab.eos)
    return Example(ids, len(prompt) if answer_only else 0)


def qa_docs(vocab: Vocab, people: Sequence[Person], answer_only: bool) -> list[Example]:
    return [
        doc(vocab, qa_prompt(p, item), [*p.value(item), "."], answer_only)
        for p in people
        for item in ITEMS
    ]


def new_model(cfg: Config, vocab: Vocab) -> GPT:
    torch.manual_seed(cfg.model_seed)
    return GPT(GPTConfig(len(vocab), cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd))


def run_train(
    cfg: Config,
    model: GPT,
    examples: list[Example],
    epochs: int,
    lr: float,
    pad: int,
    label: str,
    cosine: bool = True,
    hook: Callable[[int], None] | None = None,
) -> float:
    t0 = time.perf_counter()
    every = max(1, epochs // 5)

    def on_epoch(epoch: int, loss: float) -> None:
        if (epoch + 1) % every == 0 or epoch + 1 == epochs:
            log(
                f"[{label}] epoch {epoch + 1}/{epochs} loss {loss:.4f} {time.perf_counter() - t0:.0f}s"
            )
        if hook is not None:
            hook(epoch + 1)

    train(
        model, examples, epochs, cfg.batch_size, lr, cfg.weight_decay, pad, cfg.model_seed,
        cosine=cosine, on_epoch=on_epoch,
    )  # fmt: skip
    return time.perf_counter() - t0


def mean_qa(model: GPT, vocab: Vocab, people: Sequence[Person]) -> float:
    return float(np.mean([qa_correct(model, vocab, people, item) for item in ITEMS]))


def finetune(
    cfg: Config,
    base: GPT,
    examples: list[Example],
    vocab: Vocab,
    label: str,
    curve_people: tuple[Sequence[Person], Sequence[Person]] | None = None,
) -> tuple[GPT, dict[int, tuple[float, float]], float]:
    model = copy.deepcopy(base)
    curve: dict[int, tuple[float, float]] = {}

    def hook(epoch: int) -> None:
        if curve_people is not None and epoch in cfg.curve_epochs:
            model.eval()
            curve[epoch] = (
                mean_qa(model, vocab, curve_people[0]),
                mean_qa(model, vocab, curve_people[1]),
            )
            model.train()

    seconds = run_train(
        cfg, model, examples, cfg.finetune_epochs, cfg.finetune_lr, vocab.pad, label,
        cosine=False, hook=hook,
    )  # fmt: skip
    return model, curve, seconds


def pct(values: np.ndarray | float) -> str:
    return f"{100 * float(np.mean(values)):.1f}"


def print_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|---|" + "---:|" * (len(header) - 1))
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="knowledge storage and extraction on synthetic bios"
    )
    parser.add_argument("--quick", action="store_true", help="tiny config for a smoke run")
    args = parser.parse_args()
    cfg = QUICK if args.quick else Config()
    started = time.perf_counter()
    print(
        f"python {platform.python_version()} / torch {torch.__version__} / numpy {np.__version__}"
    )
    print(f"platform {platform.platform()} / torch threads {torch.get_num_threads()}")
    print(cfg)

    rng = np.random.default_rng(cfg.data_seed)
    people, _ = make_people(cfg.n_people, rng)
    train_idx, test_idx = split_people(cfg.n_people, rng)
    train_people = [people[i] for i in train_idx]
    test_people = [people[i] for i in test_idx]
    is_train = np.isin(np.arange(cfg.n_people), train_idx)
    vocab = Vocab()

    plain = [doc(vocab, plain_bio(p), [], False) for p in people]
    augmented = [
        doc(vocab, b, [], False) for p in people for b in augmented_bios(p, cfg.n_bios, rng)
    ]
    partial = [
        doc(vocab, b, [], False)
        for b in partially_augmented_bios(people, is_train.tolist(), cfg.n_bios, rng)
    ]
    qa_mixed = qa_docs(vocab, train_people, answer_only=False)
    qa_finetune = qa_docs(vocab, train_people, answer_only=True)
    reverse_finetune = qa_finetune + [
        doc(vocab, reverse_prompt(p), [*p.name, "."], True) for p in train_people
    ]
    print(
        f"vocab {len(vocab)} / people {len(people)} (train {len(train_people)}, test {len(test_people)})"
    )
    for name, docs in (
        ("plain", plain),
        ("augmented", augmented),
        ("partial", partial),
        ("qa", qa_mixed),
    ):
        print(f"{name}: {len(docs)} docs / {sum(len(e.ids) for e in docs)} tokens per pass")
    print(f"example plain bio: {' '.join(plain_bio(people[0]))}")
    for b in augmented_bios(people[0], 2, np.random.default_rng(1)):
        print(f"example augmented bio: {' '.join(b)}")
    print(
        f"example qa: {' '.join(qa_prompt(people[0], 'work_city'))} {people[0].value('work_city')[0]} ."
    )
    print(f"example reverse qa: {' '.join(reverse_prompt(people[0]))} {' '.join(people[0].name)} .")

    seconds: dict[str, float] = {}
    pretrain_sets = {
        "A": (plain + qa_mixed, cfg.exposures),
        "B": (plain, cfg.exposures),
        "C": (augmented, cfg.exposures // cfg.n_bios),
        "D": (partial, cfg.exposures // cfg.n_bios),
    }
    pretrained: dict[str, GPT] = {}
    for key, (docs, epochs) in pretrain_sets.items():
        pretrained[key] = new_model(cfg, vocab)
        seconds[f"{key} pretrain"] = run_train(
            cfg, pretrained[key], docs, epochs, cfg.pretrain_lr, vocab.pad, key
        )

    qa_models = {"A": pretrained["A"]}
    curves = {}
    for key in ("B", "C", "D"):
        qa_models[key], curves[key], seconds[f"{key} finetune"] = finetune(
            cfg, pretrained[key], qa_finetune, vocab, f"{key} ft", (train_people, test_people)
        )
    reverse_model, _, seconds["C reverse finetune"] = finetune(
        cfg, pretrained["C"], reverse_finetune, vocab, "C rev ft"
    )

    t0 = time.perf_counter()
    keys = tuple(qa_models)
    qa_test = {
        k: {i: qa_correct(m, vocab, test_people, i) for i in ITEMS} for k, m in qa_models.items()
    }
    qa_train = {k: mean_qa(m, vocab, train_people) for k, m in qa_models.items()}
    bio = {k: bio_correct(m, vocab, test_people, cfg.block_size - 3) for k, m in pretrained.items()}
    rev = {
        group: (
            qa_correct(reverse_model, vocab, ps, "birth_date"),
            reverse_correct(reverse_model, vocab, ps),
        )
        for group, ps in (("train", train_people), ("test", test_people))
    }
    seconds["QA and bio evaluation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    probe_keys = ("B", "C", "D")
    layers = range(1, cfg.n_layer + 1)
    probes = {
        (k, layer, hidden): {
            i: probe_correct(
                pretrained[k], vocab, train_people, test_people, i, layer, cfg.model_seed, hidden
            )
            for i in ITEMS
        }
        for hidden in (0, cfg.probe_hidden)
        for k in probe_keys
        for layer in layers
    }
    majority = {i: majority_correct(train_people, test_people, i) for i in ITEMS}
    seconds["probes"] = time.perf_counter() - t0

    print("\n## QA accuracy on test people (%)\n")
    rows = [[ITEM_LABELS[i], *(pct(qa_test[k][i]) for k in keys)] for i in ITEMS]
    rows.append(["平均", *(pct(np.concatenate(list(qa_test[k].values()))) for k in keys)])
    rows.append(["参考 訓練人物の平均", *(pct(qa_train[k]) for k in keys)])
    print_table(["項目", *keys], rows)

    print("\n## QA accuracy during finetuning, mean over items (%)\n")
    header = ["エポック"] + [f"{k} {g}" for k in curves for g in ("訓練人物", "テスト人物")]
    rows = [
        [str(e), *(pct(v) for k in curves for v in curves[k].get(e, (float("nan"),) * 2))]
        for e in cfg.curve_epochs
    ]
    print_table(header, rows)

    print("\n## bio continuation on test people, pretrained models (%)\n")
    rows = [[ITEM_LABELS[i], *(pct(bio[k][i]) for k in pretrained)] for i in ITEMS]
    rows.append(["全項目", *(pct(np.stack(list(bio[k].values())).all(axis=0)) for k in pretrained)])
    print_table(["項目", *pretrained], rows)

    for hidden, title in ((0, "linear"), (cfg.probe_hidden, f"MLP (hidden {cfg.probe_hidden})")):
        print(f"\n## {title} probe at the last name token, test people (%)\n")
        rows = [["最頻値", "", *(pct(majority[i]) for i in ITEMS)]]
        rows += [
            [k, str(layer), *(pct(probes[(k, layer, hidden)][i]) for i in ITEMS)]
            for k in probe_keys
            for layer in layers
        ]
        print_table(["モデル", "層", *(ITEM_LABELS[i] for i in ITEMS)], rows)

    print("\n## reversal (C pretrained, finetuned on both directions for train people) (%)\n")
    print_table(
        ["向き", "訓練人物", "テスト人物"],
        [
            ["順方向 名前から生年月日", pct(rev["train"][0]), pct(rev["test"][0])],
            ["逆方向 生年月日から名前", pct(rev["train"][1]), pct(rev["test"][1])],
        ],
    )

    print("\n## time (s)\n")
    for stage, sec in seconds.items():
        print(f"- {stage} {sec:.0f}")
    print(f"- total {time.perf_counter() - started:.0f}")


if __name__ == "__main__":
    main()
