"""Split source text into retrieval-friendly overlapping chunks."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Source text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of trailing characters to repeat in the next chunk.

    Returns:
        A list of ordered text chunks.

    Raises:
        ValueError: If the chunk parameters are invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
    if not text:
        return []
    # Step size is smaller than chunk_size when overlap is requested.
    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if not chunk:
            continue
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks
