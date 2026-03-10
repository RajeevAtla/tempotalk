"""Integration-style tests for retrieval output quality."""

import tomllib
from pathlib import Path

from tempus_copilot.config import load_settings
from tempus_copilot.pipeline import run_pipeline
from tests.helpers.fakes import default_retrieval_embedding_client, static_generation_client


def test_citations_map_to_chunk_ids_and_metrics_present(tmp_path: Path) -> None:
    """Ensure generated objections cite retrieved KB chunks and metrics."""
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    result = run_pipeline(
        settings,
        embedding_client=default_retrieval_embedding_client(),
        generation_client=static_generation_client(
            objection_confidence=0.9,
            script_confidence=0.8,
        ),
    )
    objections = tomllib.loads(result.objection_handlers_path.read_text(encoding="utf-8"))
    first = objections["objections"][0]
    assert first["citations"]
    assert all(str(item).startswith("product_kb.md:") for item in first["citations"])
    assert first["supporting_metrics"]


def test_pipeline_handles_empty_retrieval(tmp_path: Path) -> None:
    """Allow the pipeline to complete even when retrieval returns no hits."""
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    rag_copy = settings.rag.model_copy(update={"top_k": 0})
    settings = settings.model_copy(update={"rag": rag_copy})
    result = run_pipeline(
        settings,
        embedding_client=default_retrieval_embedding_client(),
        generation_client=static_generation_client(
            objection_confidence=0.9,
            script_confidence=0.8,
        ),
    )
    objections = tomllib.loads(result.objection_handlers_path.read_text(encoding="utf-8"))
    first = objections["objections"][0]
    assert first["citations"] == []
