from pathlib import Path

from tempus_copilot.ingest.market import load_market_intelligence


def test_load_market_intelligence_contract() -> None:
    rows = load_market_intelligence(Path("tests/fixtures/market_intelligence.csv"))
    assert len(rows) == 3
    assert rows[0].provider_id == "P001"
    assert rows[0].estimated_patient_volume == 120
