"""Streamlit GUI for the HITL web-research agent.

Run with:
    uv run streamlit run bin/web_researcher_app.py

Requires TAVILY_API_KEY and an LLM key (e.g. OPENAI_API_KEY) in the environment.
The agent pauses before every write_file; the operator approves or denies with
the APPROVE / DENY buttons, which resume the graph via Command(resume=...).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

# Allow `streamlit run bin/web_researcher_app.py` without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from langchain_core.runnables import Runnable, RunnableConfig  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.web_researcher.feedback import record_feedback  # noqa: E402
from agents.web_researcher.runner import build_web_researcher  # noqa: E402


def init_session_state() -> None:
    """Seed st.session_state keys exactly once per session."""

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("waiting_for_approval", None)
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("agent", None)
    st.session_state.setdefault("last_topic", None)
    st.session_state.setdefault("feedback_sent", False)


def reset_session() -> None:
    """Clear the conversation and start a fresh graph thread."""

    st.session_state["messages"] = []
    st.session_state["waiting_for_approval"] = None
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["last_topic"] = None
    st.session_state["feedback_sent"] = False


def _get_agent() -> Runnable[Any, Any]:
    """Build the agent once and cache it on the session."""

    if st.session_state["agent"] is None:
        st.session_state["agent"] = build_web_researcher()
    agent: Runnable[Any, Any] = st.session_state["agent"]
    return agent


def _consume_stream(agent: Runnable[Any, Any], graph_input: Any, config: RunnableConfig) -> None:
    """Drive agent.stream(stream_mode='updates') and reflect updates into the UI.

    Handles four update kinds:
    - "__interrupt__": a tool paused for approval; stash the payload and stop.
    - "agent" / "invoke_llm": the model produced an assistant message.
    - "tools" / "use_tool": a tool returned an observation.
    """

    for chunk in agent.stream(graph_input, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupts = chunk["__interrupt__"]
            st.session_state["waiting_for_approval"] = interrupts[0].value
            return

        for node, update in chunk.items():
            if node in ("agent", "invoke_llm"):
                for msg in update.get("messages", []):
                    text = _message_text(msg)
                    if text:
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": text}
                        )
            elif node in ("tools", "use_tool"):
                for msg in update.get("messages", []):
                    text = _message_text(msg)
                    if text:
                        st.session_state["messages"].append(
                            {"role": "tool", "content": text}
                        )

    st.session_state["waiting_for_approval"] = None


def run_agent(topic: str) -> None:
    """Start a research run for the given topic."""

    st.session_state["messages"].append({"role": "user", "content": topic})
    st.session_state["last_topic"] = topic
    st.session_state["feedback_sent"] = False
    agent = _get_agent()
    config: RunnableConfig = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    _consume_stream(agent, {"messages": [{"role": "user", "content": topic}]}, config)


def feedback(decision: str) -> None:
    """Resume a paused graph with the operator's APPROVE / DENY decision."""

    st.session_state["waiting_for_approval"] = None
    agent = _get_agent()
    config: RunnableConfig = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    _consume_stream(agent, Command(resume=decision), config)


def _message_text(msg: Any) -> str:
    """Extract printable text from a LangChain message or dict."""

    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, (str, dict))
        ]
        return "\n".join(p for p in parts if p)
    return ""


def _last_assistant_answer() -> str | None:
    """Return the most recent assistant message text, if any."""

    for message in reversed(st.session_state["messages"]):
        if message["role"] == "assistant" and message["content"]:
            return message["content"]
    return None


def render_feedback() -> None:
    """Render the Good/Bad feedback widget under the latest agent answer.

    Captures the 正確さ&品質 / 満足度 axes: a thumbs verdict plus an optional
    note is appended to .feedback/feedback.jsonl via record_feedback.
    """

    answer = _last_assistant_answer()
    topic = st.session_state["last_topic"]
    if answer is None or topic is None:
        return

    st.divider()
    if st.session_state["feedback_sent"]:
        st.success("フィードバックを記録しました。ありがとうございます。")
        return

    st.caption("この回答は役に立ちましたか？")
    comment = st.text_input("コメント（任意）", key="feedback_comment")
    col_good, col_bad = st.columns(2)
    if col_good.button("👍 Good"):
        record_feedback(query=topic, answer=answer, rating="good", comment=comment)
        st.session_state["feedback_sent"] = True
        st.rerun()
    if col_bad.button("👎 Bad"):
        record_feedback(query=topic, answer=answer, rating="bad", comment=comment)
        st.session_state["feedback_sent"] = True
        st.rerun()


def app() -> None:
    """Top-level Streamlit page."""

    st.set_page_config(page_title="Web Researcher (HITL)", page_icon=None)
    st.title("Web Researcher (HITL)")
    st.caption(
        "Tavily で調べ、HTML レポートを書き出す。"
        "ファイル書き込みの前に人間の承認を挟む。"
    )

    init_session_state()

    with st.sidebar:
        if st.button("New session"):
            reset_session()
            st.rerun()
        st.write(f"thread_id: `{st.session_state['thread_id']}`")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending = st.session_state["waiting_for_approval"]
    if pending is not None:
        st.warning("書き込み承認が必要です（HITL）")
        st.json(
            {
                "action": pending.get("action"),
                "path": pending.get("path"),
                "bytes": pending.get("bytes"),
            }
        )
        st.code(pending.get("preview", ""), language="html")
        col_approve, col_deny = st.columns(2)
        if col_approve.button("APPROVE", type="primary"):
            feedback("approve")
            st.rerun()
        if col_deny.button("DENY"):
            feedback("deny")
            st.rerun()
        return

    render_feedback()

    topic = st.chat_input("調べたいトピックを入力")
    if topic:
        run_agent(topic)
        st.rerun()


if __name__ == "__main__":
    app()
