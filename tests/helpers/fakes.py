from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempus_copilot.models import MeetingScriptArtifact, ObjectionArtifact


@dataclass
class KeywordEmbeddingClient:
    feature_rules: list[tuple[str, ...]]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            rows.append(
                [
                    float(any(keyword in lowered for keyword in rule))
                    for rule in self.feature_rules
                ]
            )
        return np.array(rows, dtype=np.float32)


@dataclass
class ConstantEmbeddingClient:
    dimension: int
    value: float = 1.0

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.full((len(texts), self.dimension), self.value, dtype=np.float32)


@dataclass
class EmptyKBEmbeddingClient:
    query_dimension: int = 128

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if texts and texts[0].startswith("Provider "):
            return np.zeros((len(texts), self.query_dimension), dtype=np.float32)
        return np.empty((0, 0), dtype=np.float32)


class BadShapeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)


@dataclass
class StaticGenerationClient:
    objection_confidence: float = 0.9
    script_confidence: float = 0.9
    objection_citations: list[str] | None = None
    script_citations: list[str] | None = None
    objection_prefix: str = "Objection"
    script_prefix: str = "Script"
    include_metrics: bool = True

    def generate_objection_handler(
        self,
        provider_id: str,
        concern: str,
        kb_context: str,
        citation_ids: list[str],
        observed_metrics: list[str],
    ) -> ObjectionArtifact:
        citations = citation_ids if self.objection_citations is None else self.objection_citations
        metrics = observed_metrics if self.include_metrics else []
        return ObjectionArtifact(
            provider_id=provider_id,
            concern=concern,
            response=f"{self.objection_prefix} {provider_id} {concern}",
            supporting_metrics=metrics,
            citations=citations,
            confidence=self.objection_confidence,
        )

    def generate_meeting_script(
        self,
        provider_id: str,
        tumor_focus: str,
        kb_context: str,
        citation_ids: list[str],
    ) -> MeetingScriptArtifact:
        citations = citation_ids if self.script_citations is None else self.script_citations
        return MeetingScriptArtifact(
            provider_id=provider_id,
            tumor_focus=tumor_focus,
            script=f"{self.script_prefix} {provider_id} {tumor_focus}",
            citations=citations,
            confidence=self.script_confidence,
        )


def default_retrieval_embedding_client() -> KeywordEmbeddingClient:
    return KeywordEmbeddingClient(
        feature_rules=[
            ("turnaround",),
            ("sensitivity", "specificity"),
            ("workflow", "support"),
            ("leukemia",),
        ]
    )


def pipeline_embedding_client() -> KeywordEmbeddingClient:
    return KeywordEmbeddingClient(
        feature_rules=[
            ("turnaround",),
            ("sensitivity", "specificity"),
            ("support", "workflow"),
        ]
    )


def static_generation_client(
    *,
    objection_confidence: float = 0.9,
    script_confidence: float = 0.9,
    objection_citations: list[str] | None = None,
    script_citations: list[str] | None = None,
    objection_prefix: str = "Objection",
    script_prefix: str = "Script",
    include_metrics: bool = True,
) -> StaticGenerationClient:
    return StaticGenerationClient(
        objection_confidence=objection_confidence,
        script_confidence=script_confidence,
        objection_citations=objection_citations,
        script_citations=script_citations,
        objection_prefix=objection_prefix,
        script_prefix=script_prefix,
        include_metrics=include_metrics,
    )
