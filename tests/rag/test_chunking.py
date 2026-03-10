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


def test_chunk_text_covers_continue_and_no_break_branches() -> None:
    class WeirdText:
        def __init__(self) -> None:
            self._len_calls = 0

        def __len__(self) -> int:
            # First call controls range(); later calls avoid the break condition.
            self._len_calls += 1
            return 3 if self._len_calls == 1 else 100

        def __getitem__(self, _: slice) -> str:
            return ""

    assert chunk_text(text=WeirdText(), chunk_size=2, chunk_overlap=1) == []
