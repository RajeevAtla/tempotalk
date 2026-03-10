"""Core pipeline output tests."""

import tomllib
from pathlib import Path

from tempus_copilot.config import load_settings
from tempus_copilot.output_schema import validate_run_outputs
from tempus_copilot.pipeline import run_pipeline
from tests.helpers.fakes import pipeline_embedding_client, static_generation_client


def test_pipeline_writes_toml_outputs(tmp_path: Path) -> None:
    """Verifies the pipeline writes all required TOML artifacts with expected fields."""
    settings = load_settings(Path("config/defaults.toml"))
    settings = settings.model_copy(update={"output_dir": tmp_path})
    result = run_pipeline(
        settings,
        embedding_client=pipeline_embedding_client(),
        generation_client=static_generation_client(
            objection_confidence=0.86,
            script_confidence=0.82,
        ),
    )
    assert result.ranked_providers_path.exists()
    assert result.objection_handlers_path.exists()
    assert result.meeting_scripts_path.exists()
    assert result.retrieval_debug_path.exists()

    ranked = tomllib.loads(result.ranked_providers_path.read_text(encoding="utf-8"))
    objections = tomllib.loads(result.objection_handlers_path.read_text(encoding="utf-8"))
    scripts = tomllib.loads(result.meeting_scripts_path.read_text(encoding="utf-8"))
    metadata = tomllib.loads(result.metadata_path.read_text(encoding="utf-8"))
    retrieval = tomllib.loads(result.retrieval_debug_path.read_text(encoding="utf-8"))

    assert "providers" in ranked
    assert "objections" in objections
    assert "scripts" in scripts
    assert ranked["schema_version"] == "1.0.0"
    assert objections["schema_version"] == "1.0.0"
    assert scripts["schema_version"] == "1.0.0"
    assert retrieval["schema_version"] == "1.0.0"

    first_ranked = ranked["providers"][0]
    assert "factor_scores" in first_ranked
    assert "factor_contributions" in first_ranked
    assert "calibration_terms" in first_ranked
    assert "score" in first_ranked

    first_objection = objections["objections"][0]
    assert "supporting_metrics" in first_objection
    assert "citations" in first_objection
    assert "confidence" in first_objection

    first_script = scripts["scripts"][0]
    assert "citations" in first_script
    assert "confidence" in first_script
    assert "baml_schema_sha256" in metadata
    assert "baml_prompt_sha256" in metadata
    assert retrieval["retrieval_debug"][0]["retrieved"][0]["distance"] >= 0

    errors = validate_run_outputs(result.run_dir)
    assert errors == []
