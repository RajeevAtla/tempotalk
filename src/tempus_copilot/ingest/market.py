from __future__ import annotations

from pathlib import Path

import polars as pl

from tempus_copilot.models import ProviderRecord


def load_market_intelligence(path: Path) -> list[ProviderRecord]:
    df = pl.read_csv(path)
    return [
        ProviderRecord(
            provider_id=str(row["provider_id"]),
            physician_name=str(row["physician_name"]),
            specialty=str(row["specialty"]),
            institution=str(row["institution"]),
            region=str(row["region"]),
            estimated_patient_volume=int(row["estimated_patient_volume"]),
            tumor_focus=str(row["tumor_focus"]),
            adoption_signal=float(row["adoption_signal"]),
            last_interaction_days=int(row["last_interaction_days"]),
        )
        for row in df.iter_rows(named=True)
    ]
