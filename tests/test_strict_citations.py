"""Strict citation enforcement tests."""

import tomllib
from pathlib import Path

from tempus_copilot.config import load_settings
from tempus_copilot.pipeline import run_pipeline
from tests.helpers.fakes import ConstantEmbeddingClient, static_generation_client


def test_strict_citations_sanitizes_and_reduces_confidence(tmp_path: Path) -> None:
    """Verifies strict citation mode strips invalid citations and lowers confidence."""
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    result = run_pipeline(
        settings,
        embedding_client=ConstantEmbeddingClient(dimension=4),
        generation_client=static_generation_client(
            objection_confidence=0.9,
            script_confidence=0.8,
            objection_citations=["not-allowed"],
            script_citations=["not-allowed"],
        ),
        strict_citations=True,
    )
    objections = tomllib.loads(result.objection_handlers_path.read_text(encoding="utf-8"))
    first = objections["objections"][0]
    assert first["citations"] == []
    assert float(first["confidence"]) < 0.9
