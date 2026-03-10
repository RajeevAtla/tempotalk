"""Unit tests for provider ranking heuristics."""

from tempus_copilot.models import CRMNote, ProviderRecord, RankingCalibration, RankingWeights
from tempus_copilot.ranking.score import rank_providers


def test_rank_providers_sorts_by_score_desc() -> None:
    """Sort providers from highest to lowest composite score."""
    providers = [
        ProviderRecord(
            provider_id="P001",
            physician_name="Dr A",
            specialty="Oncology",
            institution="X",
            region="NE",
            estimated_patient_volume=100,
            tumor_focus="Lung",
            adoption_signal=0.8,
            last_interaction_days=10,
        ),
        ProviderRecord(
            provider_id="P002",
            physician_name="Dr B",
            specialty="Oncology",
            institution="Y",
            region="NE",
            estimated_patient_volume=50,
            tumor_focus="Lung",
            adoption_signal=0.2,
            last_interaction_days=10,
        ),
    ]
    notes = [
        CRMNote(
            note_id="N1",
            provider_id="P001",
            timestamp="2026-02-20T10:00:00",
            concern_type="turnaround_time",
            note_text="Need confidence in turnaround time.",
            sentiment="negative",
        )
    ]
    ranked = rank_providers(
        providers=providers,
        crm_notes=notes,
        weights=RankingWeights(
            patient_volume=0.45,
            clinical_fit=0.3,
            objection_urgency=0.15,
            recency=0.1,
        ),
        calibration=RankingCalibration(
            concern_severity={
                "turnaround_time": 1.0,
                "reimbursement": 0.6,
                "evidence_strength": 0.8,
                "workflow_fit": 0.5,
                "general": 0.4,
            },
            specialty_fit={
                "Oncology": 1.0,
                "Hematology": 0.85,
                "default": 0.7,
            },
        ),
    )
    assert ranked[0].provider_id == "P001"
    assert ranked[0].score >= ranked[1].score


def test_rank_providers_applies_calibration() -> None:
    """Apply specialty and concern calibration terms to ranking."""
    providers = [
        ProviderRecord(
            provider_id="P001",
            physician_name="Dr A",
            specialty="Oncology",
            institution="X",
            region="NE",
            estimated_patient_volume=80,
            tumor_focus="Lung",
            adoption_signal=0.6,
            last_interaction_days=12,
        ),
        ProviderRecord(
            provider_id="P002",
            physician_name="Dr B",
            specialty="RareSpecialty",
            institution="Y",
            region="NE",
            estimated_patient_volume=80,
            tumor_focus="Lung",
            adoption_signal=0.6,
            last_interaction_days=12,
        ),
    ]
    notes = [
        CRMNote(
            note_id="N1",
            provider_id="P001",
            timestamp="2026-02-20T10:00:00",
            concern_type="turnaround_time",
            note_text="Need confidence in turnaround time.",
            sentiment="negative",
        ),
        CRMNote(
            note_id="N2",
            provider_id="P002",
            timestamp="2026-02-20T10:00:00",
            concern_type="workflow_fit",
            note_text="Workflow integration concerns.",
            sentiment="neutral",
        ),
    ]
    ranked = rank_providers(
        providers=providers,
        crm_notes=notes,
        weights=RankingWeights(
            patient_volume=0.25,
            clinical_fit=0.35,
            objection_urgency=0.3,
            recency=0.1,
        ),
        calibration=RankingCalibration(
            concern_severity={
                "turnaround_time": 1.0,
                "workflow_fit": 0.3,
                "general": 0.4,
            },
            specialty_fit={
                "Oncology": 1.0,
                "default": 0.5,
            },
        ),
    )
    assert ranked[0].provider_id == "P001"
