from __future__ import annotations

from typing import TypeVar

from langchain_core.output_parsers import (
    BaseOutputParser,
    JsonOutputParser,
    StrOutputParser,
)
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def string_parser() -> BaseOutputParser[str]:
    """Plain text parser; extracts the message content as a string."""

    return StrOutputParser()


def json_parser(schema: type[T]) -> JsonOutputParser:
    """Structured-output parser bound to a pydantic schema.

    The returned parser also exposes get_format_instructions() so the chain can
    inject schema hints into the system prompt.
    """

    return JsonOutputParser(pydantic_object=schema)
