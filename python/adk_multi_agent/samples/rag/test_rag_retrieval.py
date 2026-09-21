"""VertexAiRagRetrieval がモデルへのリクエストにどう付くかを固定する。

Gemini 2 以降では関数として宣言されず、Gemini 組み込みの検索としてリクエストに入る。
検索はモデル側で行われ、ツールの run_async は呼ばれない。
そのため name と description はモデルに届かず、before_tool / after_tool も効かない。

ネットワークも GCP も使わない。process_llm_request が組み立てるリクエストだけを見る。
"""

from __future__ import annotations

from google.adk.models import LlmRequest
from google.adk.tools.retrieval import VertexAiRagRetrieval
from google.genai import types

# ツール名とコーパス ID を別の文字列にする。同じにすると名前が載ったように見誤る。
SPECS = [
    ("product_docs", "製品仕様を検索する", "c1", 5),
    ("faq_docs", "よくある質問を検索する", "c2", 3),
    ("policy_docs", "返品規約を検索する", "c3", 3),
]


async def build_request(model: str) -> LlmRequest:
    request = LlmRequest(model=model, config=types.GenerateContentConfig())
    for name, description, corpus, top_k in SPECS:
        tool = VertexAiRagRetrieval(
            name=name,
            description=description,
            rag_corpora=[f"projects/p/locations/us-central1/ragCorpora/{corpus}"],
            similarity_top_k=top_k,
        )
        await tool.process_llm_request(tool_context=None, llm_request=request)
    return request


def declared_names(request: LlmRequest) -> set[str]:
    return {
        d.name
        for t in request.config.tools or []
        if t.function_declarations
        for d in t.function_declarations
    }


async def test_gemini_gets_built_in_retrieval_not_a_function() -> None:
    request = await build_request("gemini-3.8-flash")
    retrieval = [t for t in request.config.tools if t.retrieval]
    assert len(retrieval) == 3
    assert declared_names(request) == set()


async def test_tool_names_and_descriptions_never_reach_gemini() -> None:
    """Instruction で「製品の仕様は product_docs」と書いても、モデルはその名前を見ない。"""
    request = await build_request("gemini-3.8-flash")
    dumped = " ".join(str(t.model_dump()) for t in request.config.tools)
    for name, description, corpus, _ in SPECS:
        assert name not in dumped
        assert description not in dumped
        assert f"ragCorpora/{corpus}" in dumped


async def test_per_corpus_settings_survive_on_the_built_in_path() -> None:
    request = await build_request("gemini-3.8-flash")
    top_ks = [
        t.retrieval.vertex_rag_store.similarity_top_k
        for t in request.config.tools
        if t.retrieval
    ]
    assert top_ks == [5, 3, 3]


async def test_non_gemini_model_gets_named_functions() -> None:
    """Gemini 以外では関数として宣言され、名前で使い分けられる。"""
    request = await build_request("claude-sonnet-5")
    assert not [t for t in request.config.tools or [] if t.retrieval]
    assert declared_names(request) == {"product_docs", "faq_docs", "policy_docs"}
