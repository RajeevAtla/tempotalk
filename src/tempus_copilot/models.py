"""Typed domain models shared across ingestion, ranking, and output generation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ProviderRecord(BaseModel):
    """Represents one provider row from market intelligence."""

    provider_id: str
    physician_name: str
    specialty: str
    institution: str
    region: str
    estimated_patient_volume: int
    tumor_focus: str
    adoption_signal: float
    last_interaction_days: int


class CRMNote(BaseModel):
    """Represents one prior-interaction note tied to a provider."""

    note_id: str
    provider_id: str
    timestamp: str
    concern_type: str
    note_text: str
    sentiment: str


class KBDocument(BaseModel):
    """Represents one knowledge-base document after ingestion."""

    source: str
    text: str


class RankingWeights(BaseModel):
    """Stores scalar weights for provider ranking factors."""

    patient_volume: float = Field(ge=0)
    clinical_fit: float = Field(ge=0)
    objection_urgency: float = Field(ge=0)
    recency: float = Field(ge=0)


class RankingCalibration(BaseModel):
    """Stores calibration maps used by ranking heuristics."""

    concern_severity: dict[str, float]
    specialty_fit: dict[str, float]


class RankedProvider(BaseModel):
    """Captures the scored provider output written by the ranking stage."""

    provider_id: str
    physician_name: str
    institution: str
    score: float
    rationale: str
    factor_scores: dict[str, float]
    calibration_terms: dict[str, float]
    factor_contributions: dict[str, float]


class ObjectionArtifact(BaseModel):
    """Represents one generated objection handler artifact."""

    provider_id: str
    concern: str
    response: str
    supporting_metrics: list[str]
    citations: list[str]
    confidence: float = Field(ge=0, le=1)


class MeetingScriptArtifact(BaseModel):
    """Represents one generated meeting script artifact."""

    provider_id: str
    tumor_focus: str
    script: str
    citations: list[str]
    confidence: float = Field(ge=0, le=1)


class PipelineResult(BaseModel):
    """Collects the file paths produced by a pipeline run."""

    run_dir: Path
    ranked_providers_path: Path
    objection_handlers_path: Path
    meeting_scripts_path: Path
    retrieval_debug_path: Path
    metadata_path: Path
