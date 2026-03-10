"""Load CRM note exports into typed note records."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from tempus_copilot.models import CRMNote


def load_crm_notes(path: Path) -> list[CRMNote]:
    """Load CRM notes from a CSV export.

    Args:
        path: Path to the CRM notes CSV file.

    Returns:
        A list of CRM notes parsed from the CSV.
    """
    df = pl.read_csv(path)
    return [
        CRMNote(
            note_id=str(row["note_id"]),
            provider_id=str(row["provider_id"]),
            timestamp=str(row["timestamp"]),
            concern_type=str(row["concern_type"]),
            note_text=str(row["note_text"]),
            sentiment=str(row["sentiment"]),
        )
        for row in df.iter_rows(named=True)
    ]
