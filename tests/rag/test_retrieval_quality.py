import tomllib
from pathlib import Path

import numpy as np

from tempus_copilot.config import load_settings
from tempus_copilot.models import MeetingScriptArtifact, ObjectionArtifact
from tempus_copilot.pipeline import run_pipeline


class RetrievalEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            rows.append(
                [
                    float("turnaround" in lower),
                    float("sensitivity" in lower or "specificity" in lower),
                    float("workflow" in lower or "support" in lower),
                    1.0,
                ]
            )
        return np.array(rows, dtype=np.float32)


class EchoGenerationClient:
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
            response="ok",
            supporting_metrics=observed_metrics,
            citations=citation_ids,
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
            script="ok",
            citations=citation_ids,
            confidence=0.8,
        )


def test_citations_map_to_chunk_ids_and_metrics_present(tmp_path: Path) -> None:
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    result = run_pipeline(
        settings,
        embedding_client=RetrievalEmbeddingClient(),
        generation_client=EchoGenerationClient(),
    )
    objections = tomllib.loads(result.objection_handlers_path.read_text(encoding="utf-8"))
    first = objections["objections"][0]
    assert first["citations"]
    assert all(str(item).startswith("product_kb.md:") for item in first["citations"])
    assert first["supporting_metrics"]


def test_pipeline_handles_empty_retrieval(tmp_path: Path) -> None:
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    rag_copy = settings.rag.model_copy(update={"top_k": 0})
    settings = settings.model_copy(update={"rag": rag_copy})
    result = run_pipeline(
        settings,
        embedding_client=RetrievalEmbeddingClient(),
        generation_client=EchoGenerationClient(),
    )
    objections = tomllib.loads(result.objection_handlers_path.read_text(encoding="utf-8"))
    first = objections["objections"][0]
    assert first["citations"] == []
