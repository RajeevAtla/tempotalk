"""Unit tests for chunking behavior and edge cases."""

import pytest

from tempus_copilot.rag import chunking as chunking_module
from tempus_copilot.rag.chunking import chunk_text


def test_chunk_text_respects_size() -> None:
    """Split large text into bounded chunks."""
    text = "A" * 1200
    chunks = chunk_text(text=text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_chunk_text_validation_and_empty_input() -> None:
    """Reject invalid chunk settings and handle empty input."""
    with pytest.raises(ValueError):
        chunk_text(text="abc", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        chunk_text(text="abc", chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError):
        chunk_text(text="abc", chunk_size=10, chunk_overlap=10)
    assert chunk_text(text="", chunk_size=10, chunk_overlap=1) == []


def test_chunk_text_handles_short_and_exact_boundaries() -> None:
    """Keep short or exact inputs in a single chunk."""
    assert chunk_text(text="ab", chunk_size=2, chunk_overlap=1) == ["ab"]
    assert chunk_text(text="abcd", chunk_size=4, chunk_overlap=1) == ["abcd"]


def test_chunk_text_preserves_overlap_between_chunks() -> None:
    """Retain the configured overlap between adjacent chunks."""
    chunks = chunk_text(text="abcdefghij", chunk_size=4, chunk_overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij"]


def test_chunk_text_breaks_after_final_partial_chunk() -> None:
    """Stop after emitting the final partial chunk."""
    chunks = chunk_text(text="abcdefgh", chunk_size=5, chunk_overlap=1)
    assert chunks == ["abcde", "efgh"]


def test_chunk_text_handles_empty_chunk_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover the defensive branch that skips empty chunks."""
    monkeypatch.setattr(
        chunking_module,
        "range",
        lambda start, stop, step: [stop],
        raising=False,
    )
    assert chunk_text(text="abc", chunk_size=2, chunk_overlap=1) == []
