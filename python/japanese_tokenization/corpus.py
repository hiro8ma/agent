from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
KENPOU_PATH = PROJECT_DIR.parents[1] / "go" / "internal" / "enginecmp" / "kenpou.json"
TEXTS_DIR = PROJECT_DIR / "texts"
TRAINING_ARTICLES = 80


@dataclass(frozen=True)
class Document:
    name: str
    text: str


SHORT_SENTENCES: list[tuple[str, str]] = [
    ("教材の例", "カレーライスはおいしい"),
    ("教材の例", "こんにちは"),
    ("教材の例", "仏滅にはカレーを食べない"),
    ("カタカナが多い", "マイクロサービスのコンテナイメージをレジストリにプッシュする"),
    ("漢字が多い", "年次有給休暇取得促進施策検討会議事録"),
    ("英数字が混ざる", "Pod の CPU 使用率が 80% を超えたら HPA で replicas を 3 から 10 に増やす"),
    ("英語", "Curry and rice is delicious."),
]


def load_kenpou() -> list[str]:
    articles: list[str] = json.loads(KENPOU_PATH.read_text(encoding="utf-8"))
    return articles


def training_text() -> str:
    return "\n".join(load_kenpou()[:TRAINING_ARTICLES])


def training_lines() -> list[str]:
    return [line for line in training_text().split("\n") if line]


def documents() -> list[Document]:
    articles = load_kenpou()
    return [
        Document(f"憲法 第1〜{TRAINING_ARTICLES}条", "\n".join(articles[:TRAINING_ARTICLES])),
        Document(
            f"憲法 第{TRAINING_ARTICLES + 1}〜{len(articles)}条",
            "\n".join(articles[TRAINING_ARTICLES:]),
        ),
        Document("吾輩は猫である 冒頭", (TEXTS_DIR / "wagahai.txt").read_text(encoding="utf-8")),
        Document("技術文書の例", (TEXTS_DIR / "tech_note.txt").read_text(encoding="utf-8")),
    ]
