from hashlib import sha256
from pathlib import Path

import pytest

from tempus_copilot import pipeline as pipeline_module
from tempus_copilot.config import load_settings
from tempus_copilot.pipeline import run_pipeline
from tests.helpers.fakes import (
    BadShapeEmbeddingClient,
    EmptyKBEmbeddingClient,
    static_generation_client,
)


def test_pipeline_ensure_inputs_generates_mock_data_for_missing_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "generated_data"
    output_dir = tmp_path / "out"
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={
            "market_csv": data_dir / "market_intelligence.csv",
            "crm_csv": data_dir / "crm_notes.csv",
            "kb_markdown": data_dir / "product_kb.md",
            "output_dir": output_dir,
        }
    )
    called = {"value": False}

    def fake_generate(output_dir: Path, seed: int, scale: int) -> None:
        called["value"] = True
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "market_intelligence.csv").write_text(
            (
                "provider_id,physician_name,specialty,institution,region,"
                "estimated_patient_volume,tumor_focus,adoption_signal,last_interaction_days\n"
                "P001,Dr A,Oncology,X,NE,100,Lung,0.8,10\n"
            ),
            encoding="utf-8",
        )
        (output_dir / "crm_notes.csv").write_text(
            (
                "note_id,provider_id,timestamp,concern_type,note_text,sentiment\n"
                "N001,P001,2026-02-20T09:30:00,turnaround_time,Need faster TAT,negative\n"
            ),
            encoding="utf-8",
        )
        (output_dir / "product_kb.md").write_text(
            "# KB\nAverage turnaround is 8 days.", encoding="utf-8"
        )

    monkeypatch.setattr(pipeline_module, "generate_mock_data", fake_generate)
    result = run_pipeline(
        settings,
        embedding_client=EmptyKBEmbeddingClient(),
        generation_client=static_generation_client(),
    )
    assert called["value"] is True
    assert result.ranked_providers_path.exists()


def test_compute_baml_hashes_uses_empty_hash_when_source_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = Path("this/does/not/exist.baml")
    monkeypatch.setattr(pipeline_module, "_baml_source_path", lambda: missing)
    schema_hash, prompt_hash = pipeline_module._compute_baml_hashes()
    expected = sha256(b"").hexdigest()
    assert schema_hash == expected
    assert prompt_hash == expected


def test_extract_metrics_deduplicates_values() -> None:
    metrics = pipeline_module._extract_metrics("8 days in 8 days with 99.1% and 99.1%")
    assert metrics.count("8") == 1
    assert metrics.count("99.1") == 1


def test_default_embedding_client_returns_ollama_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings(Path("config/defaults.toml"))
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "https://ollama.com")
    client = pipeline_module._default_embedding_client(settings)
    assert isinstance(client, pipeline_module.OllamaEmbeddingClient)

    non_ollama = settings.model_copy(
        update={
            "models": settings.models.model_copy(update={"embedding_provider": "google"}),
        }
    )
    with pytest.raises(ValueError):
        pipeline_module._default_embedding_client(non_ollama)


def test_pipeline_handles_empty_kb_embedding_matrix(tmp_path: Path) -> None:
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    result = run_pipeline(
        settings,
        embedding_client=EmptyKBEmbeddingClient(),
        generation_client=static_generation_client(),
    )
    assert result.metadata_path.exists()


def test_pipeline_rejects_non_2d_embedding_matrix(tmp_path: Path) -> None:
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    with pytest.raises(ValueError):
        run_pipeline(
            settings,
            embedding_client=BadShapeEmbeddingClient(),
            generation_client=static_generation_client(),
        )


def test_pipeline_low_confidence_threshold_non_violation(tmp_path: Path) -> None:
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    result = run_pipeline(
        settings,
        embedding_client=EmptyKBEmbeddingClient(),
        generation_client=static_generation_client(),
        fail_on_low_confidence=0.2,
    )
    assert result.run_dir.exists()
