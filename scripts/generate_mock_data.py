"""Generates deterministic mock input files for local runs and tests."""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl

SPECIALTIES = ["Oncology", "Hematology", "Thoracic Oncology"]
TUMOR_FOCUS = ["Lung", "Breast", "Leukemia", "Colorectal", "Melanoma"]
CONCERNS = ["turnaround_time", "reimbursement", "evidence_strength", "workflow_fit"]


def _make_market_rows(scale: int, rng: random.Random) -> list[dict[str, object]]:
    """Builds synthetic market intelligence rows.

    Args:
        scale: Number of providers to create.
        rng: Seeded random generator.

    Returns:
        Mock market rows.
    """
    rows: list[dict[str, object]] = []
    for i in range(scale):
        idx = i + 1
        rows.append(
            {
                "provider_id": f"P{idx:03d}",
                "physician_name": f"Dr. Provider {idx}",
                "specialty": rng.choice(SPECIALTIES),
                "institution": f"Institution {rng.randint(1, 12)}",
                "region": rng.choice(["Northeast", "Midwest", "South", "West"]),
                "estimated_patient_volume": rng.randint(40, 220),
                "tumor_focus": rng.choice(TUMOR_FOCUS),
                "adoption_signal": round(rng.uniform(0.2, 0.95), 2),
                "last_interaction_days": rng.randint(1, 60),
            }
        )
    return rows


def _make_crm_rows(scale: int, rng: random.Random) -> list[dict[str, object]]:
    """Builds synthetic CRM note rows.

    Args:
        scale: Number of note rows to create.
        rng: Seeded random generator.

    Returns:
        Mock CRM rows.
    """
    rows: list[dict[str, object]] = []
    for i in range(scale):
        provider_id = f"P{i + 1:03d}"
        concern = rng.choice(CONCERNS)
        rows.append(
            {
                "note_id": f"N{i + 1:03d}",
                "provider_id": provider_id,
                "timestamp": f"2026-02-{(i % 27) + 1:02d}T09:00:00",
                "concern_type": concern,
                "note_text": (
                    f"Discussion with {provider_id}; "
                    f"concern about {concern.replace('_', ' ')}."
                ),
                "sentiment": rng.choice(["negative", "neutral", "positive"]),
            }
        )
    return rows


def _kb_markdown() -> str:
    """Builds the small deterministic markdown knowledge base.

    Returns:
        Markdown text for the local KB fixture.
    """
    # Keep the KB tiny and deterministic so local runs and tests stay reproducible.
    return """# Tempus Product Knowledge

## Turnaround Time
Average standard oncology panel turnaround is 8 calendar days.

## Analytical Performance
Recent validation shows 99.1% sensitivity and 99.4% specificity.

## Clinical Workflow
EMR integration available with 24-hour support response.
"""


def generate_mock_data(output_dir: Path, seed: int, scale: int) -> None:
    """Writes deterministic mock input files to disk.

    Args:
        output_dir: Target directory for the generated files.
        seed: Seed used for reproducible pseudo-random values.
        scale: Number of mock provider and CRM rows to generate.
    """
    # Market and CRM rows share one seeded RNG so fixture generation is stable across runs.
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    market = pl.DataFrame(_make_market_rows(scale=scale, rng=rng))
    crm = pl.DataFrame(_make_crm_rows(scale=scale, rng=rng))
    market.write_csv(output_dir / "market_intelligence.csv")
    crm.write_csv(output_dir / "crm_notes.csv")
    (output_dir / "product_kb.md").write_text(_kb_markdown(), encoding="utf-8")


def main() -> None:
    """Runs the script with repo-default arguments."""
    generate_mock_data(output_dir=Path("data/mock"), seed=42, scale=25)


if __name__ == "__main__":
    main()
