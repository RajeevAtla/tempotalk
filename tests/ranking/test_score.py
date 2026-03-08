from tempus_copilot.models import CRMNote, ProviderRecord, RankingWeights
from tempus_copilot.ranking.score import rank_providers


def test_rank_providers_sorts_by_score_desc() -> None:
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
    )
    assert ranked[0].provider_id == "P001"
    assert ranked[0].score >= ranked[1].score
