"""Hybrid RAG retrieval tools for the helpdesk agent.

Two retrievers cover complementary query shapes:

- `search_manual` — keyword search (Elasticsearch-style). Best for exact matches on
  product numbers, error codes, and domain jargon found in manuals / release notes.
- `search_qa` — vector search (Qdrant-style). Best for semantic similarity against
  past question / answer pairs where the wording differs but the intent matches.

Both default to an in-memory fake backend so the graph runs with zero infrastructure.
Set `HELPDESK_BACKEND=real` to swap in the Elasticsearch / Qdrant connectors instead;
those are imported lazily so the import of this module never requires the clients.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from langchain_core.tools import StructuredTool


@dataclass(frozen=True)
class Document:
    """A single retrieved passage with its source label and relevance score."""

    source: str
    text: str
    score: float


class Retriever(Protocol):
    """Common shape for keyword and vector backends, real or fake."""

    def search(self, query: str, top_k: int) -> list[Document]: ...


# --- Sample corpus (generic, fictional system "XYZ"; no real product data) ---

_MANUAL_CORPUS: list[tuple[str, str]] = [
    ("manual:XYZ-1001", "Error code E-204 on system XYZ means the session token expired. "
     "Re-authenticate from the admin console to refresh the token."),
    ("manual:XYZ-1002", "Release note for XYZ v3.2: the export endpoint now requires the "
     "scope `data:export`. Requests without it return HTTP 403."),
    ("manual:XYZ-1003", "To rotate an API key for system XYZ, open Settings > Credentials "
     "and select Rotate. The previous key stays valid for 24 hours."),
    ("manual:XYZ-1004", "Error code E-500 indicates an internal batch job failure. Check the "
     "job queue dashboard and retry the failed job."),
]

_QA_CORPUS: list[tuple[str, str]] = [
    ("qa:0001", "Q: My login keeps getting kicked out. A: Your token likely expired; "
     "sign in again and the session will be restored."),
    ("qa:0002", "Q: Downloads are failing with a permission error. A: Ask an admin to grant "
     "the export permission to your account."),
    ("qa:0003", "Q: How do I change my access key safely? A: Use the rotate option in "
     "settings; the old key works for a day so nothing breaks."),
    ("qa:0004", "Q: The nightly process didn't run. A: A background job failed; rerun it "
     "from the dashboard or contact support if it repeats."),
]


# 同義語辞書。専門用語と固有名詞の揺れを吸収する。
#
# 教材が「形態素解析の工夫」で挙げる要素。
# 「APIキー」と「アクセスキー」が別語のままだと、
# どちらか一方でしか引けない。
_SYNONYMS: dict[str, set[str]] = {
    "api": {"apiキー", "アクセスキー", "認証キー"},
    "key": {"キー", "鍵"},
    "rotate": {"ローテーション", "更新", "再発行"},
    "error": {"エラー", "障害"},
    "token": {"トークン"},
    "session": {"セッション"},
    "export": {"エクスポート", "書き出し"},
    "login": {"ログイン", "サインイン", "ログインが切れる", "ログイン切れ"},
    "expired": {"切れる", "期限切れ", "失効"},
    "permission": {"権限", "パーミッション"},
    "job": {"ジョブ", "バッチ"},
}
# 逆引き。日本語から英語の見出し語へ寄せる。
_SYNONYM_INDEX: dict[str, str] = {
    alias.lower(): head for head, aliases in _SYNONYMS.items() for alias in aliases
}

# 検索に寄与しない語。落とさないと、どの文書とも当たってしまう。
#
# 実際、"How do I configure the quantum flux capacitor" が
# how / do / the で無関係な文書に score=1 で当たっていた。
_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "how", "what", "i", "my",
    "to", "of", "in", "on", "for", "and", "or", "it", "this", "that",
    "教えて", "ください", "について", "とは", "です", "ます",
}


def _tokenize(text: str) -> set[str]:
    """語に分ける。英数字と日本語の両方を扱う。

    元は `[a-z0-9-]+` だけを拾っていた。
    英語のコーパスに日本語で問い合わせると 1 トークンも残らず、
    常に「該当なし」になる。エラーは出ないので、
    検索が動いていないことに気づけない。

    日本語は形態素解析器を入れるのが本筋だが、
    依存を増やさずに済ませるため 2-gram で近似する。
    実運用では MeCab や Sudachi にカスタム辞書を足す。
    """

    lowered = text.lower()
    tokens = {t for t in re.findall(r"[a-z0-9-]+", lowered) if t not in _STOPWORDS}

    # 日本語（ひらがな・カタカナ・漢字）の連続を取り出す。
    for run in re.findall(r"[ぁ-んァ-ヴー一-龥]+", lowered):
        if run in _STOPWORDS:
            continue
        # 2 文字以上なら 2-gram、1 文字ならそのまま。
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
        tokens.add(run)

    # 同義語を見出し語へ寄せる。
    expanded = set(tokens)
    for t in tokens:
        head = _SYNONYM_INDEX.get(t)
        if head:
            expanded.add(head)
    return expanded


class FakeKeywordRetriever:
    """Token-overlap keyword search. Stand-in for an Elasticsearch BM25 query."""

    def __init__(self, corpus: list[tuple[str, str]]) -> None:
        self._corpus = corpus

    def search(self, query: str, top_k: int) -> list[Document]:
        q_tokens = _tokenize(query)
        scored: list[Document] = []
        for source, text in self._corpus:
            overlap = len(q_tokens & _tokenize(text))
            if overlap:
                scored.append(Document(source=source, text=text, score=float(overlap)))
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]


def _question_part(text: str) -> str:
    """QA 形式のテキストから質問部分だけを取り出す。

    教材の「ベクトルの工夫」にあたる。
    質問と回答をまとめてベクトル化すると、
    回答側の語が距離を押し広げる。
    ユーザーが投げるのは質問なので、質問文どうしを近づけたほうが当たる。
    """

    body = text.split("A:", 1)[0]
    return body.replace("Q:", "").strip() or text


class FakeVectorRetriever:
    """Jaccard-similarity search. Stand-in for a Qdrant cosine-similarity query.

    Jaccard over token sets is a deterministic, dependency-free proxy for semantic
    similarity, enough to demonstrate vector-style retrieval offline.

    索引に載せるのは質問部分だけにする。返す本文は Q と A の両方を含む。
    """

    def __init__(self, corpus: list[tuple[str, str]], index_question_only: bool = True) -> None:
        self._corpus = corpus
        self._index_question_only = index_question_only

    def search(self, query: str, top_k: int) -> list[Document]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scored: list[Document] = []
        for source, text in self._corpus:
            indexed = _question_part(text) if self._index_question_only else text
            d_tokens = _tokenize(indexed)
            union = q_tokens | d_tokens
            if not union:
                continue
            similarity = len(q_tokens & d_tokens) / len(union)
            if similarity > 0:
                scored.append(Document(source=source, text=text, score=round(similarity, 3)))
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]


def _build_keyword_retriever() -> Retriever:
    if os.environ.get("HELPDESK_BACKEND") == "real":
        from .connectors import build_elasticsearch_retriever

        return build_elasticsearch_retriever()
    return FakeKeywordRetriever(_MANUAL_CORPUS)


def _build_vector_retriever() -> Retriever:
    if os.environ.get("HELPDESK_BACKEND") == "real":
        from .connectors import build_qdrant_retriever

        return build_qdrant_retriever()
    return FakeVectorRetriever(_QA_CORPUS)


def _rrf(rankings: list[list[Document]], k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion で複数の検索結果を統合する。

    教材が言う「ハイブリッド検索」の実装。
    スコアの尺度が違う検索器（キーワードの重なり数と Jaccard 係数）を
    そのまま足すと、片方の尺度に引きずられる。
    順位だけを使えば尺度に依存しない。
    """

    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc.source] = scores.get(doc.source, 0.0) + 1.0 / (k + rank)
            docs.setdefault(doc.source, doc)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        Document(source=src, text=docs[src].text, score=round(score, 5))
        for src, score in ordered
    ]


def _format(docs: list[Document]) -> str:
    if not docs:
        return "(no matching documents)"
    return "\n".join(f"[{d.source} score={d.score}] {d.text}" for d in docs)


def search_manual(query: str, top_k: int = 3) -> str:
    """Keyword search over product manuals and release notes (Elasticsearch-style).

    Use this for exact-match lookups: error codes (e.g. E-204), product numbers,
    version strings, API scope names, or other precise domain terms.
    """

    return _format(_build_keyword_retriever().search(query, top_k))


def search_qa(query: str, top_k: int = 3) -> str:
    """Vector search over past question / answer pairs (Qdrant-style).

    Use this for semantic lookups where the user describes a problem in their own
    words and you want similar past cases rather than an exact term match.
    """

    return _format(_build_vector_retriever().search(query, top_k))


SEARCH_MANUAL_TOOL = StructuredTool.from_function(
    func=search_manual,
    name="search_manual",
    description=(
        "Keyword search over manuals and release notes. Prefer for exact terms: "
        "error codes, product numbers, versions, API scopes."
    ),
)

SEARCH_QA_TOOL = StructuredTool.from_function(
    func=search_qa,
    name="search_qa",
    description=(
        "Vector search over past Q&A. Prefer for paraphrased, symptom-style questions "
        "where semantic similarity beats exact keyword match."
    ),
)

ALL_TOOLS = [SEARCH_MANUAL_TOOL, SEARCH_QA_TOOL]


def search_hybrid(query: str, top_k: int = 3) -> str:
    """キーワード検索とベクトル検索を統合して引く。

    教材は検索手法が不適切なときの解決策として
    「ハイブリッド検索（ベクトル検索とキーワード検索の適切な組み合わせ）」を挙げる。

    既存の 2 つは**選択**であって組み合わせではなかった。
    ツール選択で片方に倒すと、選び損ねた側にしか無い文書に到達できない。
    """

    keyword = _build_keyword_retriever().search(query, top_k * 2)
    vector = _build_vector_retriever().search(query, top_k * 2)
    return _format(_rrf([keyword, vector])[:top_k])
