"""用途ごとの Compaction の設定。

Compaction には発火点が 2 つある。

    compaction_interval  Invocation の終了後に、未圧縮の Invocation の数で判定する
    token_threshold      各モデル呼び出しの直前と、Invocation の終了後の両方で判定する

token_threshold の判定には、直前のモデル応答が申告した prompt_token_count を使う。
圧縮してもその値は古いまま残るので、次の応答が低い値を申告するまで判定のたびに圧縮が走る。
毎回「前回の要約 + 新しいイベント」を要約し直すので、情報の損失は積み重なる。

要約器を渡さないと、ルートエージェント自身のモデルで要約する。
本体に高いモデルを使っていると、要約も同じ単価になるので、ここでは明示的に渡せる形にしている。
"""

from __future__ import annotations

from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import BaseLlm, Gemini

# (compaction_interval, overlap_size)。教材の用途別の値。
# overlap_size の分は前回の範囲と重ねて要約し直すので、その分だけ要約のコストが増える。
PRESETS: dict[str, tuple[int, int]] = {
    "chatbot": (10, 1),
    "support": (20, 2),
    "research": (30, 3),
    "codegen": (50, 5),
    "task": (30, 2),
}


def compaction_config(
    use_case: str, summarizer_model: BaseLlm | str | None = None
) -> EventsCompactionConfig:
    """用途に合わせた Compaction の設定を返す。

    summarizer_model を渡さない場合は、ADK の既定どおりルートエージェントのモデルで要約する。
    """
    if use_case not in PRESETS:
        raise ValueError(f"未知の用途: {use_case!r}（{', '.join(PRESETS)} のいずれか）")
    interval, overlap = PRESETS[use_case]
    summarizer = None
    if summarizer_model is not None:
        llm = (
            Gemini(model=summarizer_model)
            if isinstance(summarizer_model, str)
            else summarizer_model
        )
        summarizer = LlmEventSummarizer(llm=llm)
    return EventsCompactionConfig(
        compaction_interval=interval, overlap_size=overlap, summarizer=summarizer
    )
