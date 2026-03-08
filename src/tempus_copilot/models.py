from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ProviderRecord(BaseModel):
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
    note_id: str
    provider_id: str
    timestamp: str
    concern_type: str
    note_text: str
    sentiment: str


class KBDocument(BaseModel):
    source: str
    text: str


class RankingWeights(BaseModel):
    patient_volume: float = Field(ge=0)
    clinical_fit: float = Field(ge=0)
    objection_urgency: float = Field(ge=0)
    recency: float = Field(ge=0)


class RankingCalibration(BaseModel):
    concern_severity: dict[str, float]
    specialty_fit: dict[str, float]


class RankedProvider(BaseModel):
    provider_id: str
    physician_name: str
    institution: str
    score: float
    rationale: str
    factor_scores: dict[str, float]


class PipelineResult(BaseModel):
    run_dir: Path
    ranked_providers_path: Path
    objection_handlers_path: Path
    meeting_scripts_path: Path
    metadata_path: Path
