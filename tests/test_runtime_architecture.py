from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tempus_copilot import pipeline as pipeline_module
from tempus_copilot.config import load_settings
from tempus_copilot.pipeline import run_pipeline
from tests.helpers.fakes import EmptyKBEmbeddingClient, static_generation_client


def test_pipeline_uses_handwritten_generation_adapter_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    expected_client = static_generation_client()
    calls = {"count": 0}

    def fake_default_generation_client(**_: object):
        calls["count"] += 1
        return expected_client

    monkeypatch.setattr(
        pipeline_module,
        "get_default_generation_client",
        fake_default_generation_client,
    )
    run_pipeline(
        settings,
        embedding_client=EmptyKBEmbeddingClient(),
    )
    assert calls["count"] == 1


def test_pipeline_resolves_both_default_runtime_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    expected_embedder = EmptyKBEmbeddingClient()
    expected_generator = static_generation_client()
    calls = {"embedder": 0, "generator": 0}

    def fake_default_embedding_client(settings_obj):
        calls["embedder"] += 1
        assert settings_obj is settings
        return expected_embedder

    def fake_default_generation_client(**kwargs: object):
        calls["generator"] += 1
        assert kwargs["generation_model"] == settings.models.generation_model
        return expected_generator

    monkeypatch.setattr(
        pipeline_module,
        "_default_embedding_client",
        fake_default_embedding_client,
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_default_generation_client",
        fake_default_generation_client,
    )
    result = run_pipeline(settings)
    assert result.metadata_path.exists()
    assert calls == {"embedder": 1, "generator": 1}


def test_pipeline_does_not_import_baml_client_for_runtime_execution(
    tmp_path: Path,
) -> None:
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={"output_dir": tmp_path}
    )
    sys.modules.pop("baml_client", None)
    run_pipeline(
        settings,
        embedding_client=EmptyKBEmbeddingClient(),
        generation_client=static_generation_client(),
    )
    assert "baml_client" not in sys.modules
