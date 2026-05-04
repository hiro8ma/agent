from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    StringPromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableLambda

# PromptTemplate for the user turn. Kept as a separate object so callers can
# inspect / reuse the template independently of the chat-level wiring.
QUESTION_TEMPLATE: PromptTemplate = PromptTemplate.from_template(
    "Question:\n{question}\n\n"
    "Context (extracted from the PDF, ordered by appearance):\n{context}\n\n"
    "Answer the question using only the context above."
)


class _SystemPromptTemplate(StringPromptTemplate):
    """Concrete StringPromptTemplate appending parser format instructions.

    Lives at module scope (not nested) so pydantic's runtime model construction
    can resolve the class without a closure over local variables.
    """

    template: str = ""
    format_instructions: str = ""

    def format(self, **kwargs: Any) -> str:
        if self.format_instructions:
            return f"{self.template}\n\n{self.format_instructions}"
        return self.template


def build_chain(
    provider: BaseChatModel,
    system_prompt: str,
    parser: BaseOutputParser[Any],
) -> Runnable[dict[str, Any], Any]:
    """Compose an LCEL chain: prompt -> chat model -> output parser.

    Input shape: {"question": str, "context": str, "chat_history": list[BaseMessage]}
    Output: whatever the parser returns (str for StrOutputParser, dict for JsonOutputParser).
    """

    format_instructions = ""
    get_fi = getattr(parser, "get_format_instructions", None)
    if callable(get_fi):
        try:
            format_instructions = get_fi()
        except (NotImplementedError, AttributeError):
            format_instructions = ""

    system_template = _SystemPromptTemplate(
        input_variables=[],
        template=system_prompt,
        format_instructions=format_instructions,
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template.format()),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", QUESTION_TEMPLATE.template),
        ]
    )

    # Ensure chat_history defaults to an empty list so callers can omit it.
    def _ensure_history(payload: dict[str, Any]) -> dict[str, Any]:
        if "chat_history" not in payload or payload["chat_history"] is None:
            payload = {**payload, "chat_history": []}
        return payload

    pre: Runnable[dict[str, Any], dict[str, Any]] = RunnableLambda(_ensure_history)

    return pre | chat_prompt | provider | parser


def seed_history(turns: list[tuple[str, str]]) -> list[BaseMessage]:
    """Convert (role, content) tuples into LangChain Message objects.

    Accepted roles: 'system', 'user'/'human', 'assistant'/'ai'. Useful when a
    caller wants to prime the MessagesPlaceholder slot before invoking a chain.
    """

    messages: list[BaseMessage] = []
    for role, content in turns:
        normalized = role.lower()
        if normalized == "system":
            messages.append(SystemMessage(content=content))
        elif normalized in ("user", "human"):
            messages.append(HumanMessage(content=content))
        elif normalized in ("assistant", "ai"):
            messages.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unsupported role: {role!r}")
    return messages
