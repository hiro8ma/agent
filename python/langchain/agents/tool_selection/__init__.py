"""tool_selection — three reusable tool-selection strategies over one tool catalog.

- native: bind all tools, 1 LLM call (small stable catalogs)
- semantic: embed + top-k, then bind k, 1 LLM call (large catalogs)
- hierarchical: group router, 2 LLM calls (large categorized catalogs)
"""

from . import hierarchical, native, semantic
from .catalog import ALL_TOOLS, GROUPS
from .llm import Selection

__all__ = ["ALL_TOOLS", "GROUPS", "Selection", "hierarchical", "native", "semantic"]
