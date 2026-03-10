from pathlib import Path

from tempus_copilot.ingest import kb as kb_module
from tempus_copilot.ingest.kb import load_kb_markdown, load_kb_pdf


def test_load_kb_markdown_contract() -> None:
    docs = load_kb_markdown(Path("tests/fixtures/product_kb.md"))
    assert len(docs) == 1
    assert "turnaround" in docs[0].text.lower()


def test_load_kb_pdf_contract_with_mocked_reader(monkeypatch) -> None:
    class FakePage:
        def __init__(self, value: str | None) -> None:
            self._value = value

        def extract_text(self) -> str | None:
            return self._value

    class FakeReader:
        def __init__(self, _: str) -> None:
            self.pages = [FakePage("First page"), FakePage(None), FakePage("Third page")]

    monkeypatch.setattr(kb_module, "PdfReader", FakeReader)
    docs = load_kb_pdf(Path("fake.pdf"))
    assert docs[0].source == "fake.pdf"
    assert docs[0].text == "First page\n\nThird page"
