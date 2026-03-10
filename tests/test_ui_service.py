from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w

from tempus_copilot.config import load_settings
from tempus_copilot.ui_service import (
    RunControls,
    SettingsOverride,
    apply_settings_overrides,
    discover_run_dirs,
    list_artifact_files,
    load_run_summary,
    load_ui_settings,
    load_validation_report,
    most_recent_run_dir,
    run_pipeline_from_ui,
    validate_run_summary,
)
from tests.helpers.output_builders import write_valid_run_dir


def test_load_ui_settings_uses_config_loader() -> None:
    settings = load_ui_settings(Path("config/defaults.toml"))
    expected = load_settings(Path("config/defaults.toml"))
    assert settings == expected


def test_apply_settings_overrides_updates_nested_models() -> None:
    settings = load_settings(Path("config/defaults.toml"))
    updated = apply_settings_overrides(
        settings,
        SettingsOverride(
            market_csv=Path("custom/market.csv"),
            generation_model="custom-gen",
            embedding_model="custom-embed",
            chunk_size=256,
            top_k=9,
            strict_citations=True,
            patient_volume_weight=0.5,
        ),
    )
    assert updated.market_csv == Path("custom/market.csv")
    assert updated.models.generation_model == "custom-gen"
    assert updated.models.embedding_model == "custom-embed"
    assert updated.rag.chunk_size == 256
    assert updated.rag.top_k == 9
    assert updated.output.strict_citations is True
    assert updated.ranking_weights.patient_volume == 0.5


def test_run_pipeline_from_ui_applies_overrides_and_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    settings = load_settings(Path("config/defaults.toml"))

    def fake_load(config_path: Path) -> object:
        calls["config_path"] = config_path
        return settings

    class _Result:
        run_dir = Path("outputs/run_test")

    def fake_run_pipeline(
        run_settings: object,
        *,
        strict_citations: object,
        fail_on_low_confidence: object,
    ) -> object:
        calls["settings"] = run_settings
        calls["strict_citations"] = strict_citations
        calls["fail_on_low_confidence"] = fail_on_low_confidence
        return _Result()

    monkeypatch.setattr("tempus_copilot.ui_service.load_ui_settings", fake_load)
    monkeypatch.setattr("tempus_copilot.ui_service.run_pipeline", fake_run_pipeline)

    run_dir = run_pipeline_from_ui(
        Path("config/custom.toml"),
        settings_overrides=SettingsOverride(top_k=7),
        controls=RunControls(strict_citations=True, fail_on_low_confidence=0.6),
    )

    updated = calls["settings"]
    assert isinstance(updated, type(settings))
    assert updated.rag.top_k == 7
    assert calls["config_path"] == Path("config/custom.toml")
    assert calls["strict_citations"] is True
    assert calls["fail_on_low_confidence"] == 0.6
    assert run_dir == Path("outputs/run_test")


def test_discover_run_dirs_returns_sorted_run_directories(tmp_path: Path) -> None:
    (tmp_path / "run_20260310_010000").mkdir()
    (tmp_path / "run_20260309_235959").mkdir()
    (tmp_path / "notes").mkdir()
    runs = discover_run_dirs(tmp_path)
    assert runs == [
        tmp_path / "run_20260310_010000",
        tmp_path / "run_20260309_235959",
    ]


def test_most_recent_run_dir_returns_latest_run(tmp_path: Path) -> None:
    (tmp_path / "run_20260310_010000").mkdir()
    (tmp_path / "run_20260309_235959").mkdir()
    assert most_recent_run_dir(tmp_path) == tmp_path / "run_20260310_010000"


def test_validate_run_summary_returns_errors_for_invalid_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_invalid"
    run_dir.mkdir()
    summary = validate_run_summary(run_dir)
    assert summary.run_dir == run_dir
    assert not summary.is_valid
    assert summary.errors


def test_load_run_summary_parses_outputs_into_display_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260310_101010"
    write_valid_run_dir(run_dir)
    ranked = {
        "schema_version": "1.0.0",
        "providers": [
            {
                "provider_id": "P001",
                "physician_name": "Dr Example",
                "institution": "North Hospital",
                "score": 0.91,
                "rationale": "high fit",
            }
        ],
    }
    objections = {
        "schema_version": "1.0.0",
        "objections": [
            {
                "provider_id": "P001",
                "concern": "turnaround_time",
                "response": "Use the 8-day turnaround data.",
                "supporting_metrics": ["8 days"],
                "citations": ["product_kb.md:0"],
                "confidence": 0.83,
            }
        ],
    }
    scripts = {
        "schema_version": "1.0.0",
        "scripts": [
            {
                "provider_id": "P001",
                "tumor_focus": "Lung",
                "script": "Open with lung panel value.",
                "citations": ["product_kb.md:0"],
                "confidence": 0.79,
            }
        ],
    }
    retrieval = {
        "schema_version": "1.0.0",
        "retrieval_debug": [
            {
                "provider_id": "P001",
                "query_text": "Provider P001 with tumor focus Lung.",
                "retrieved": [
                    {
                        "chunk_id": "product_kb.md:0",
                        "source": "product_kb.md",
                        "distance": 0.12,
                    }
                ],
            }
        ],
    }
    metadata = {
        "schema_version": "1.0.0",
        "output_checksum_sha256": "abc",
        "baml_schema_sha256": "schema",
        "baml_prompt_sha256": "prompt",
        "provider_count": 1,
    }
    (run_dir / "ranked_providers.toml").write_bytes(tomli_w.dumps(ranked).encode("utf-8"))
    (run_dir / "objection_handlers.toml").write_bytes(tomli_w.dumps(objections).encode("utf-8"))
    (run_dir / "meeting_scripts.toml").write_bytes(tomli_w.dumps(scripts).encode("utf-8"))
    (run_dir / "retrieval_debug.toml").write_bytes(tomli_w.dumps(retrieval).encode("utf-8"))
    (run_dir / "run_metadata.toml").write_bytes(tomli_w.dumps(metadata).encode("utf-8"))

    summary = load_run_summary(run_dir)

    assert summary.run_dir == run_dir
    assert summary.providers[0]["provider_id"] == "P001"
    assert summary.objections[0]["supporting_metrics"] == ["8 days"]
    assert summary.meeting_scripts[0]["tumor_focus"] == "Lung"
    assert summary.retrieval_debug[0]["retrieved"][0]["chunk_id"] == "product_kb.md:0"
    assert summary.metadata["provider_count"] == 1
    assert not summary.validation.is_valid
    assert summary.validation.errors == ["Checksum mismatch between metadata and outputs"]


def test_load_validation_report_groups_file_statuses_and_checksum(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260310_121212"
    write_valid_run_dir(run_dir)
    metadata = {
        "schema_version": "1.0.0",
        "output_checksum_sha256": "mismatch",
        "baml_schema_sha256": "schema",
        "baml_prompt_sha256": "prompt",
    }
    (run_dir / "run_metadata.toml").write_bytes(tomli_w.dumps(metadata).encode("utf-8"))

    report = load_validation_report(run_dir)

    assert report.run_dir == run_dir
    assert report.checksum_error == "Checksum mismatch between metadata and outputs"
    assert report.file_statuses[0].label == "Ranked Providers"
    assert report.file_statuses[0].is_valid
    assert not report.is_valid


def test_list_artifact_files_returns_expected_download_entries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_20260310_101010"
    write_valid_run_dir(run_dir)

    artifacts = list_artifact_files(run_dir)

    assert artifacts[0].label == "Ranked Providers"
    assert artifacts[0].exists
    assert artifacts[0].download_name == "run_20260310_101010_ranked_providers.toml"


def test_load_run_summary_handles_missing_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_empty"
    run_dir.mkdir()
    summary = load_run_summary(run_dir)
    assert summary.providers == []
    assert summary.objections == []
    assert summary.meeting_scripts == []
    assert summary.retrieval_debug == []
    assert summary.metadata == {}
    assert not summary.validation.is_valid
