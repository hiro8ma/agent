"""Strategy 3 — hierarchical LLM router.

Tools are organized into groups (Computation / Automation / Communication). Routing is
two LLM passes:

    layer 1: pick the group from group descriptions (LLM sees 3 groups)
    layer 2: pick the tool within that group (LLM sees ~2 tools)
    then: decide parameters and execute

Cost: 2 LLM calls. Scale: best for very large catalogs that fan out into clear
categories — each layer sees a tiny slice, so token cost stays flat as the catalog grows
within groups. Degrades when a query spans groups or the group taxonomy is wrong: a bad
layer-1 pick is unrecoverable at layer 2. That cascade risk is the WHY for keeping groups
disjoint and well-named.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage, SystemMessage

from .catalog import GROUPS, GROUPS_BY_NAME, run_tool
from .llm import Selection, fake_args, fake_pick_tool

_GROUP_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("solve", "equation", "convert", "calculate", "math", "=", "unit"), "Computation"),
    (("webhook", "workflow", "trigger", "schedule", "cron", "job", "automation"), "Automation"),
    (("slack", "channel", "#", "notify", "message", "email", "mail", "post"), "Communication"),
]


def _fake_pick_group(query: str) -> str:
    lowered = query.lower()
    for terms, group in _GROUP_HINTS:
        if any(t in lowered for t in terms):
            return group
    return GROUPS[0].name


def _llm_pick_group(query: str) -> str:
    from core.providers.factory import select_provider

    catalog = "\n".join(f"- {g.name}: {g.description}" for g in GROUPS)
    model = select_provider()
    msg = model.invoke(
        [
            SystemMessage(
                content=(
                    "Pick exactly one tool group for the user request. "
                    f"Reply with only the group name.\nGroups:\n{catalog}"
                )
            ),
            HumanMessage(content=query),
        ]
    )
    reply = str(msg.content).strip()
    for g in GROUPS:
        if g.name.lower() in reply.lower():
            return g.name
    return _fake_pick_group(query)


def _llm_pick_tool(query: str, group_name: str) -> str:
    from core.providers.factory import select_provider

    from .catalog import TOOLS_BY_NAME

    group = GROUPS_BY_NAME[group_name]
    listing = "\n".join(
        f"- {n}: {TOOLS_BY_NAME[n].description}" for n in group.tool_names
    )
    model = select_provider()
    msg = model.invoke(
        [
            SystemMessage(
                content=(
                    "Pick exactly one tool from this group for the request. "
                    f"Reply with only the tool name.\nTools:\n{listing}"
                )
            ),
            HumanMessage(content=query),
        ]
    )
    reply = str(msg.content).strip()
    for n in group.tool_names:
        if n in reply:
            return n
    return fake_pick_tool(query, list(group.tool_names))


def select(query: str, use_fake: bool = False) -> Selection:
    is_fake = use_fake or not os.environ.get("OPENAI_API_KEY")

    if is_fake:
        group_name = _fake_pick_group(query)
        candidates = list(GROUPS_BY_NAME[group_name].tool_names)
        tool_name = fake_pick_tool(query, candidates)
    else:
        group_name = _llm_pick_group(query)
        candidates = list(GROUPS_BY_NAME[group_name].tool_names)
        tool_name = _llm_pick_tool(query, group_name)

    args = fake_args(query, tool_name)
    return Selection(
        strategy="hierarchical",
        tool_name=tool_name,
        args=args,
        llm_calls=2,
        considered=[group_name, *candidates],
        result=run_tool(tool_name, args),
        is_fake=is_fake,
    )
