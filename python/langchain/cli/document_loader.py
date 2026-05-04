from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


def load_pdf(path: str) -> list[Document]:
    """Load a PDF into LangChain Documents.

    Only absolute paths are accepted to avoid ambiguous CWD-relative resolution
    when this is invoked from CI or other agents.
    """

    if not os.path.isabs(path):
        raise ValueError(f"PDF path must be absolute: {path!r}")

    resolved = Path(path).resolve(strict=False)
    if not resolved.exists():
        raise FileNotFoundError(f"PDF not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"PDF path is not a file: {resolved}")
    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {resolved.suffix!r}")

    loader: BaseLoader = PyPDFLoader(str(resolved))
    return loader.load()
