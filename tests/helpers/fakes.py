"""Reusable fake clients for pipeline and retrieval tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempus_copilot.models import MeetingScriptArtifact, ObjectionArtifact


@dataclass
class KeywordEmbeddingClient:
    """Create simple keyword-based embedding vectors for deterministic tests."""

    feature_rules: list[tuple[str, ...]]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts by marking whether each feature rule matches.

        Args:
            texts: Input texts to encode.

        Returns:
            A dense float32 matrix with one column per feature rule.
        """
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
    """Return a constant embedding for each input row."""

    dimension: int
    value: float = 1.0

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed each text with the configured constant value."""
        return np.full((len(texts), self.dimension), self.value, dtype=np.float32)


@dataclass
class EmptyKBEmbeddingClient:
    """Simulate empty-KB embeddings while still supporting provider queries."""

    query_dimension: int = 128

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return query vectors for provider prompts and empty matrices otherwise."""
        if texts and texts[0].startswith("Provider "):
            return np.zeros((len(texts), self.query_dimension), dtype=np.float32)
        return np.empty((0, 0), dtype=np.float32)


class BadShapeEmbeddingClient:
    """Return an invalid embedding shape for negative-path tests."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return a one-dimensional array to trigger shape validation."""
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)


@dataclass
class StaticGenerationClient:
    """Return deterministic generation artifacts without calling a model."""

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
        """Build a fixed objection artifact for the requested provider."""
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
        """Build a fixed meeting script artifact for the requested provider."""
        citations = citation_ids if self.script_citations is None else self.script_citations
        return MeetingScriptArtifact(
            provider_id=provider_id,
            tumor_focus=tumor_focus,
            script=f"{self.script_prefix} {provider_id} {tumor_focus}",
            citations=citations,
            confidence=self.script_confidence,
        )


def default_retrieval_embedding_client() -> KeywordEmbeddingClient:
    """Create the default keyword embedder used by retrieval-focused tests."""
    return KeywordEmbeddingClient(
        feature_rules=[
            ("turnaround",),
            ("sensitivity", "specificity"),
            ("workflow", "support"),
            ("leukemia",),
        ]
    )


def pipeline_embedding_client() -> KeywordEmbeddingClient:
    """Create the keyword embedder used by end-to-end pipeline tests."""
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
    """Create a configurable deterministic generation client for tests."""
    return StaticGenerationClient(
        objection_confidence=objection_confidence,
        script_confidence=script_confidence,
        objection_citations=objection_citations,
        script_citations=script_citations,
        objection_prefix=objection_prefix,
        script_prefix=script_prefix,
        include_metrics=include_metrics,
    )
