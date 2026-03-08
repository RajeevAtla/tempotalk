from pathlib import Path

from tempus_copilot.ingest.kb import load_kb_markdown


def test_load_kb_markdown_contract() -> None:
    docs = load_kb_markdown(Path("tests/fixtures/product_kb.md"))
    assert len(docs) == 1
    assert "turnaround" in docs[0].text.lower()
