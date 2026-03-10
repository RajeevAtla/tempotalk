"""Contract tests for knowledge-base ingestion."""

from pathlib import Path

from tempus_copilot.ingest import kb as kb_module
from tempus_copilot.ingest.kb import load_kb_markdown, load_kb_pdf


def test_load_kb_markdown_contract() -> None:
    """Load markdown KB content into a single document."""
    docs = load_kb_markdown(Path("tests/fixtures/product_kb.md"))
    assert len(docs) == 1
    assert "turnaround" in docs[0].text.lower()


def test_load_kb_pdf_contract_with_mocked_reader(monkeypatch) -> None:
    """Load PDF KB content while skipping empty page extracts."""

    class FakePage:
        """Mimic a PDF page with optional extracted text."""

        def __init__(self, value: str | None) -> None:
            """Store the fake page text."""
            self._value = value

        def extract_text(self) -> str | None:
            """Return the fake extracted page text."""
            return self._value

    class FakeReader:
        """Mimic a PDF reader with a fixed page sequence."""

        def __init__(self, _: str) -> None:
            """Populate the fake pages list."""
            self.pages = [FakePage("First page"), FakePage(None), FakePage("Third page")]

    monkeypatch.setattr(kb_module, "PdfReader", FakeReader)
    docs = load_kb_pdf(Path("fake.pdf"))
    assert docs[0].source == "fake.pdf"
    assert docs[0].text == "First page\n\nThird page"
