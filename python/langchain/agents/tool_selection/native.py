"""Strategy 1 — standard native bind.

Bind every tool to the LLM and let one tool-calling pass choose. This is the default
LangChain pattern (``llm.bind_tools(ALL_TOOLS)``).

Cost: 1 LLM call. Scale: best for a small, stable tool set. Degrades as the catalog
grows — every tool's full schema is sent on every call, inflating tokens and giving the
model more chances to mispick. That degradation is the WHY for strategies 2 and 3.
"""

from __future__ import annotations

import os

from .catalog import ALL_TOOLS, run_tool
from .llm import Selection, fake_args, fake_pick_tool


def select(query: str, use_fake: bool = False) -> Selection:
    names = [t.name for t in ALL_TOOLS]
    is_fake = use_fake or not os.environ.get("OPENAI_API_KEY")

    if not is_fake:
        from core.providers.factory import select_provider

        model = select_provider().bind_tools(ALL_TOOLS)
        msg = model.invoke(query)
        calls = getattr(msg, "tool_calls", []) or []
        if calls:
            tool_name = str(calls[0]["name"])
            args = dict(calls[0]["args"])
        else:
            tool_name = fake_pick_tool(query, names)
            args = fake_args(query, tool_name)
    else:
        tool_name = fake_pick_tool(query, names)
        args = fake_args(query, tool_name)

    return Selection(
        strategy="native",
        tool_name=tool_name,
        args=args,
        llm_calls=1,
        considered=names,
        result=run_tool(tool_name, args),
        is_fake=is_fake,
    )
