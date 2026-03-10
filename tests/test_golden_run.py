"""Golden output regression tests."""

from pathlib import Path

from tempus_copilot.config import load_settings
from tempus_copilot.pipeline import run_pipeline
from tests.helpers.fakes import default_retrieval_embedding_client, static_generation_client


def test_golden_run_outputs_match_fixtures(tmp_path: Path) -> None:
    """Verifies pipeline outputs remain byte-for-byte aligned with golden fixtures."""
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={
            "market_csv": Path("tests/fixtures/market_intelligence.csv"),
            "crm_csv": Path("tests/fixtures/crm_notes.csv"),
            "kb_markdown": Path("tests/fixtures/product_kb.md"),
            "output_dir": tmp_path,
        }
    )
    result = run_pipeline(
        settings,
        embedding_client=default_retrieval_embedding_client(),
        generation_client=static_generation_client(
            objection_confidence=0.77,
            script_confidence=0.74,
        ),
        strict_citations=True,
    )
    assert (
        result.ranked_providers_path.read_text(encoding="utf-8")
        == Path("tests/fixtures/golden/ranked_providers.toml").read_text(encoding="utf-8")
    )
    assert (
        result.objection_handlers_path.read_text(encoding="utf-8")
        == Path("tests/fixtures/golden/objection_handlers.toml").read_text(encoding="utf-8")
    )
    assert (
        result.meeting_scripts_path.read_text(encoding="utf-8")
        == Path("tests/fixtures/golden/meeting_scripts.toml").read_text(encoding="utf-8")
    )
    assert result.retrieval_debug_path.exists()
