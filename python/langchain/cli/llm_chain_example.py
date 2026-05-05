from __future__ import annotations

import os
import sys

# legacy: LLMChain is deprecated; prefer LCEL (`prompt | llm | parser`) or
# langgraph.prebuilt.create_react_agent. Kept here as a teaching artefact.
# In langchain v1 the legacy chains live in the langchain-classic package.
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from core.providers.factory import select_provider


def build_legacy_chain() -> LLMChain:
    """Construct the same prompt-LLM pairing that LCEL expresses as `prompt | llm`."""

    prompt = PromptTemplate.from_template(
        "Summarize the following sentence in one short clause:\n{sentence}"
    )
    return LLMChain(llm=select_provider(), prompt=prompt)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set; build_legacy_chain() will raise on call.",
            file=sys.stderr,
        )
        return 1

    chain = build_legacy_chain()
    sentence = (
        "LangChain provides a large catalogue of integrations for building "
        "LLM-powered applications across many providers."
    )
    result = chain.invoke({"sentence": sentence})
    print(result.get("text", result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
