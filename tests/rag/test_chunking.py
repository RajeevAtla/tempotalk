import pytest

from tempus_copilot.rag.chunking import chunk_text


def test_chunk_text_respects_size() -> None:
    text = "A" * 1200
    chunks = chunk_text(text=text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_chunk_text_validation_and_empty_input() -> None:
    with pytest.raises(ValueError):
        chunk_text(text="abc", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        chunk_text(text="abc", chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError):
        chunk_text(text="abc", chunk_size=10, chunk_overlap=10)
    assert chunk_text(text="", chunk_size=10, chunk_overlap=1) == []


def test_chunk_text_handles_short_and_exact_boundaries() -> None:
    assert chunk_text(text="ab", chunk_size=2, chunk_overlap=1) == ["ab"]
    assert chunk_text(text="abcd", chunk_size=4, chunk_overlap=1) == ["abcd"]
