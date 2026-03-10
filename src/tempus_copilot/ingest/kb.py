"""Load product knowledge documents from markdown or PDF sources."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from tempus_copilot.models import KBDocument


def load_kb_markdown(path: Path) -> list[KBDocument]:
    """Load a markdown knowledge-base document.

    Args:
        path: Path to the markdown knowledge-base file.

    Returns:
        A single-document list containing the markdown contents.
    """
    return [KBDocument(source=path.name, text=path.read_text(encoding="utf-8"))]


def load_kb_pdf(path: Path) -> list[KBDocument]:
    """Load a PDF knowledge-base document.

    Args:
        path: Path to the PDF knowledge-base file.

    Returns:
        A single-document list containing the concatenated page text.
    """
    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    return [KBDocument(source=path.name, text="\n".join(chunks))]
