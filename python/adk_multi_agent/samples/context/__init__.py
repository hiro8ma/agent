"""コンテキスト予算と汚染対策。"""

from .budget import (
    TRIM_REPORT_KEY,
    ContextBudget,
    content_tokens,
    drop_stale_tool_results,
    enforce_budget,
    rough_token_estimate,
    trim_to_limit,
)

__all__ = [
    "TRIM_REPORT_KEY",
    "ContextBudget",
    "content_tokens",
    "drop_stale_tool_results",
    "enforce_budget",
    "rough_token_estimate",
    "trim_to_limit",
]
