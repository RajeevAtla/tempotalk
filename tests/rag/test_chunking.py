from tempus_copilot.rag.chunking import chunk_text


def test_chunk_text_respects_size() -> None:
    text = "A" * 1200
    chunks = chunk_text(text=text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 500 for chunk in chunks)
