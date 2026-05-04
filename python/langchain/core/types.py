from __future__ import annotations

from pydantic import BaseModel, Field


class RunInput(BaseModel):
    """Input for a single agent run."""

    pdf_path: str = Field(..., description="Absolute path to the source PDF.")
    question: str = Field(..., description="User question to answer from the PDF.")
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=2000)
    max_chunks: int = Field(
        default=20,
        ge=1,
        description="Upper bound on chunks fed into the prompt (simple stuff strategy).",
    )


class RunOutput(BaseModel):
    """Result of a single agent run."""

    answer: str
    used_chunks: int
    total_chunks: int


class StructuredAnswer(BaseModel):
    """Schema used by JsonOutputParser when callers want structured output."""

    answer: str = Field(..., description="Direct answer to the question.")
    bullets: list[str] = Field(
        default_factory=list, description="Supporting points, bulleted."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Page numbers or chunk identifiers backing the answer.",
    )
