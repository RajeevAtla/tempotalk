import tomllib
from pathlib import Path

import numpy as np

from tempus_copilot.config import load_settings
from tempus_copilot.models import MeetingScriptArtifact, ObjectionArtifact
from tempus_copilot.pipeline import run_pipeline


class SimpleEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


class MaliciousCitationClient:
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
            response="response",
            supporting_metrics=observed_metrics,
            citations=["not-allowed"],
            confidence=0.9,
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
            script="script",
            citations=["not-allowed"],
            confidence=0.8,
        )


def test_strict_citations_sanitizes_and_reduces_confidence(tmp_path: Path) -> None:
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    result = run_pipeline(
        settings,
        embedding_client=SimpleEmbeddingClient(),
        generation_client=MaliciousCitationClient(),
        strict_citations=True,
    )
    objections = tomllib.loads(result.objection_handlers_path.read_text(encoding="utf-8"))
    first = objections["objections"][0]
    assert first["citations"] == []
    assert float(first["confidence"]) < 0.9
