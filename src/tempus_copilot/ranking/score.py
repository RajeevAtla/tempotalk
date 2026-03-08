from __future__ import annotations

from collections import defaultdict

from tempus_copilot.models import (
    CRMNote,
    ProviderRecord,
    RankedProvider,
    RankingCalibration,
    RankingWeights,
)


def rank_providers(
    providers: list[ProviderRecord],
    crm_notes: list[CRMNote],
    weights: RankingWeights,
    calibration: RankingCalibration,
) -> list[RankedProvider]:
    note_counts: dict[str, int] = defaultdict(int)
    note_severity_totals: dict[str, float] = defaultdict(float)
    for note in crm_notes:
        note_counts[note.provider_id] += 1
        severity = calibration.concern_severity.get(
            note.concern_type,
            calibration.concern_severity.get("general", 0.5),
        )
        note_severity_totals[note.provider_id] += severity

    max_volume = max((provider.estimated_patient_volume for provider in providers), default=1)
    ranked: list[RankedProvider] = []
    for provider in providers:
        volume_score = provider.estimated_patient_volume / max_volume
        specialty_fit = calibration.specialty_fit.get(
            provider.specialty,
            calibration.specialty_fit.get("default", 0.7),
        )
        fit_score = provider.adoption_signal * specialty_fit
        note_count = max(note_counts[provider.provider_id], 1)
        avg_severity = note_severity_totals[provider.provider_id] / note_count
        objection_score = min(avg_severity * (note_counts[provider.provider_id] / 2.0), 1.0)
        recency_score = 1.0 / (1.0 + float(provider.last_interaction_days))
        score = (
            volume_score * weights.patient_volume
            + fit_score * weights.clinical_fit
            + objection_score * weights.objection_urgency
            + recency_score * weights.recency
        )
        factor_scores = {
            "patient_volume": volume_score,
            "clinical_fit": fit_score,
            "objection_urgency": objection_score,
            "recency": recency_score,
        }
        ranked.append(
            RankedProvider(
                provider_id=provider.provider_id,
                physician_name=provider.physician_name,
                institution=provider.institution,
                score=score,
                rationale=(
                    f"volume={volume_score:.2f}, fit={fit_score:.2f}, "
                    f"objections={objection_score:.2f}, recency={recency_score:.2f}"
                ),
                factor_scores=factor_scores,
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked
