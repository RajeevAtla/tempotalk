from pathlib import Path

from tempus_copilot.ingest.crm import load_crm_notes


def test_load_crm_notes_contract() -> None:
    notes = load_crm_notes(Path("tests/fixtures/crm_notes.csv"))
    assert len(notes) == 3
    assert notes[0].concern_type == "turnaround_time"
