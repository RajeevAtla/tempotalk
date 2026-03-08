from __future__ import annotations

from collections import defaultdict

from tempus_copilot.models import CRMNote, ProviderRecord, RankedProvider, RankingWeights


def rank_providers(
    providers: list[ProviderRecord],
    crm_notes: list[CRMNote],
    weights: RankingWeights,
) -> list[RankedProvider]:
    note_counts: dict[str, int] = defaultdict(int)
    for note in crm_notes:
        note_counts[note.provider_id] += 1

    max_volume = max((provider.estimated_patient_volume for provider in providers), default=1)
    ranked: list[RankedProvider] = []
    for provider in providers:
        volume_score = provider.estimated_patient_volume / max_volume
        fit_score = provider.adoption_signal
        objection_score = min(note_counts[provider.provider_id] / 3.0, 1.0)
        recency_score = 1.0 / (1.0 + float(provider.last_interaction_days))
        score = (
            volume_score * weights.patient_volume
            + fit_score * weights.clinical_fit
            + objection_score * weights.objection_urgency
            + recency_score * weights.recency
        )
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
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked
