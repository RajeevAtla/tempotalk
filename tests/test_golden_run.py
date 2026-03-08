from pathlib import Path

import numpy as np

from tempus_copilot.config import load_settings
from tempus_copilot.models import MeetingScriptArtifact, ObjectionArtifact
from tempus_copilot.pipeline import run_pipeline


class GoldenEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            rows.append(
                [
                    float("turnaround" in lowered),
                    float("sensitivity" in lowered or "specificity" in lowered),
                    float("workflow" in lowered or "support" in lowered),
                    float("leukemia" in lowered),
                ]
            )
        return np.array(rows, dtype=np.float32)


class GoldenGenerationClient:
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
            response=f"Objection {provider_id} {concern}",
            supporting_metrics=observed_metrics,
            citations=citation_ids,
            confidence=0.77,
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
            script=f"Script {provider_id} {tumor_focus}",
            citations=citation_ids,
            confidence=0.74,
        )


def test_golden_run_outputs_match_fixtures(tmp_path: Path) -> None:
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
        embedding_client=GoldenEmbeddingClient(),
        generation_client=GoldenGenerationClient(),
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
