from pathlib import Path

from tempus_copilot.config import load_settings


def test_load_settings_reads_toml() -> None:
    settings = load_settings(Path("config/defaults.toml"))
    assert settings.models.generation_model == "gemini-2.5-flash"
    assert settings.rag.top_k == 4
    assert settings.ranking_weights.patient_volume > settings.ranking_weights.recency
    assert settings.ranking_calibration.concern_severity["turnaround_time"] == 1.0
