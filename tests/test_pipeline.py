import tomllib
from pathlib import Path

import numpy as np

from tempus_copilot.config import load_settings
from tempus_copilot.models import MeetingScriptArtifact, ObjectionArtifact
from tempus_copilot.output_schema import validate_run_outputs
from tempus_copilot.pipeline import run_pipeline


class FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("turnaround" in lowered),
                    float("sensitivity" in lowered or "specificity" in lowered),
                    float("support" in lowered or "workflow" in lowered),
                ]
            )
        return np.array(vectors, dtype=np.float32)


class FakeGenerationClient:
    def generate_objection_handler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
        citation_ids: list[str],
        observed_metrics: list[str],
    ) -> ObjectionArtifact:
        return ObjectionArtifact(
            provider_id=provider_id,
            concern=concern,
            response=f"{provider_id}:{concern}:{kb_context[:40]}",
            supporting_metrics=observed_metrics,
            citations=citation_ids,
            confidence=0.86,
        )

    def generate_meeting_script(
        self,
        provider_id: str,
        tumor_focus: str,
        kb_context: str,
        citation_ids: list[str],
    ) -> MeetingScriptArtifact:
        return MeetingScriptArtifact(
            provider_id=provider_id,
            tumor_focus=tumor_focus,
            script=f"{provider_id}:{tumor_focus}:{kb_context[:40]}",
            citations=citation_ids,
            confidence=0.82,
        )


def test_pipeline_writes_toml_outputs(tmp_path: Path) -> None:
    settings = load_settings(Path("config/defaults.toml"))
    settings = settings.model_copy(update={"output_dir": tmp_path})
    result = run_pipeline(
        settings,
        embedding_client=FakeEmbeddingClient(),
        generation_client=FakeGenerationClient(),
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
