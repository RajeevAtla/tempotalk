"""Tests for ui service."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w

from tempus_copilot.config import load_settings
from tempus_copilot.ui_service import (
    ArtifactFileView,
    RunControls,
    RunSummary,
    SettingsOverride,
    ValidationFileStatus,
    ValidationReport,
    apply_settings_overrides,
    discover_run_dirs,
    list_artifact_files,
    load_run_bundle,
    load_run_summary,
    load_ui_settings,
    load_validation_report,
    most_recent_run_dir,
    run_pipeline_from_ui,
    validate_run_summary,
)
from tests.helpers.output_builders import write_valid_run_dir


def test_load_ui_settings_uses_config_loader() -> None:
    """Test load ui settings uses config loader.
    """
    settings = load_ui_settings(Path("config/defaults.toml"))
    expected = load_settings(Path("config/defaults.toml"))
    assert settings == expected


def test_load_default_settings_uses_same_loader() -> None:
    """Test load default settings uses same loader.
    """
    from tempus_copilot.ui_service import load_default_settings

    assert load_default_settings(Path("config/defaults.toml")) == load_settings(
        Path("config/defaults.toml")
    )


def test_apply_settings_overrides_updates_nested_models() -> None:
    """Test apply settings overrides updates nested models.
    """
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


def test_apply_settings_overrides_updates_all_supported_fields() -> None:
    """Test apply settings overrides updates all supported fields.
    """
    settings = load_settings(Path("config/defaults.toml"))
    updated = apply_settings_overrides(
        settings,
        SettingsOverride(
            crm_csv=Path("custom/crm.csv"),
            kb_markdown=Path("custom/kb.md"),
            output_dir=Path("custom/out"),
            chunk_overlap=12,
            request_retries=4,
            backoff_seconds=1.5,
            clinical_fit_weight=0.25,
            objection_urgency_weight=0.35,
            recency_weight=0.4,
        ),
    )

    assert updated.crm_csv == Path("custom/crm.csv")
    assert updated.kb_markdown == Path("custom/kb.md")
    assert updated.output_dir == Path("custom/out")
    assert updated.rag.chunk_overlap == 12
    assert updated.rag.request_retries == 4
    assert updated.rag.backoff_seconds == 1.5
    assert updated.ranking_weights.clinical_fit == 0.25
    assert updated.ranking_weights.objection_urgency == 0.35
    assert updated.ranking_weights.recency == 0.4


def test_run_pipeline_from_ui_applies_overrides_and_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test run pipeline from ui applies overrides and controls.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    calls: dict[str, object] = {}
    settings = load_settings(Path("config/defaults.toml"))

    def fake_load(config_path: Path) -> object:
        """Fake load.
        
        Args:
            config_path: Filesystem path for config path.
        
        Returns:
            Computed result.
        """
        calls["config_path"] = config_path
        return settings

    class _Result:
        """Result."""
        run_dir = Path("outputs/run_test")

    def fake_run_pipeline(
        run_settings: object,
        *,
        strict_citations: object,
        fail_on_low_confidence: object,
    ) -> object:
        """Fake run pipeline.
        
        Args:
            run_settings: Run settings.
            strict_citations: Strict citations.
            fail_on_low_confidence: Fail on low confidence.
        
        Returns:
            Computed result.
        """
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


def test_run_pipeline_from_ui_uses_default_controls_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test run pipeline from ui uses default controls when none.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    settings = load_settings(Path("config/defaults.toml"))

    class ResultStub:
        """Result stub."""
        run_dir = Path("outputs/run_default")

    captured: dict[str, object] = {}

    monkeypatch.setattr("tempus_copilot.ui_service.load_ui_settings", lambda path: settings)

    def fake_run_pipeline(
        run_settings: object,
        *,
        strict_citations: object,
        fail_on_low_confidence: object,
    ) -> ResultStub:
        """Fake run pipeline.
        
        Args:
            run_settings: Run settings.
            strict_citations: Strict citations.
            fail_on_low_confidence: Fail on low confidence.
        
        Returns:
            Computed result.
        """
        captured["settings"] = run_settings
        captured["strict_citations"] = strict_citations
        captured["fail_on_low_confidence"] = fail_on_low_confidence
        return ResultStub()

    monkeypatch.setattr("tempus_copilot.ui_service.run_pipeline", fake_run_pipeline)

    run_dir = run_pipeline_from_ui()

    assert captured["settings"] == settings
    assert captured["strict_citations"] is None
    assert captured["fail_on_low_confidence"] is None
    assert run_dir == Path("outputs/run_default")


def test_apply_settings_overrides_noop_returns_same_values() -> None:
    """Test apply settings overrides noop returns same values.
    """
    settings = load_settings(Path("config/defaults.toml"))

    updated = apply_settings_overrides(settings, SettingsOverride())

    assert updated == settings


def test_discover_run_dirs_returns_sorted_run_directories(tmp_path: Path) -> None:
    """Test discover run dirs returns sorted run directories.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    (tmp_path / "run_20260310_010000").mkdir()
    (tmp_path / "run_20260309_235959").mkdir()
    (tmp_path / "notes").mkdir()
    runs = discover_run_dirs(tmp_path)
    assert runs == [
        tmp_path / "run_20260310_010000",
        tmp_path / "run_20260309_235959",
    ]


def test_most_recent_run_dir_returns_latest_run(tmp_path: Path) -> None:
    """Test most recent run dir returns latest run.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    (tmp_path / "run_20260310_010000").mkdir()
    (tmp_path / "run_20260309_235959").mkdir()
    assert most_recent_run_dir(tmp_path) == tmp_path / "run_20260310_010000"


def test_discover_and_recent_run_handle_missing_output_dir(tmp_path: Path) -> None:
    """Test discover and recent run handle missing output dir.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    missing = tmp_path / "missing"
    assert discover_run_dirs(missing) == []
    assert most_recent_run_dir(missing) is None


def test_validate_run_summary_returns_errors_for_invalid_run(tmp_path: Path) -> None:
    """Test validate run summary returns errors for invalid run.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    run_dir = tmp_path / "run_invalid"
    run_dir.mkdir()
    summary = validate_run_summary(run_dir)
    assert summary.run_dir == run_dir
    assert not summary.is_valid
    assert summary.errors


def test_load_run_summary_parses_outputs_into_display_rows(tmp_path: Path) -> None:
    """Test load run summary parses outputs into display rows.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
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
    """Test load validation report groups file statuses and checksum.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
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


def test_load_validation_report_marks_missing_files_and_keys(tmp_path: Path) -> None:
    """Test load validation report marks missing files and keys.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    from tempus_copilot import ui_service

    run_dir = tmp_path / "run_20260310_131313"
    run_dir.mkdir()
    write_valid_run_dir(run_dir)
    (run_dir / "meeting_scripts.toml").unlink()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ui_service,
        "validate_run_outputs",
        lambda path: [
            "ranked_providers.toml missing key: providers",
            "Missing output file: meeting_scripts.toml",
        ],
    )

    try:
        report = load_validation_report(run_dir)
    finally:
        monkeypatch.undo()

    provider_status = next(
        status for status in report.file_statuses if status.file_name == "ranked_providers.toml"
    )
    script_status = next(
        status for status in report.file_statuses if status.file_name == "meeting_scripts.toml"
    )
    assert not provider_status.is_valid
    assert provider_status.errors == ["ranked_providers.toml missing key: providers"]
    assert not script_status.exists
    assert script_status.errors == ["Missing output file: meeting_scripts.toml"]


def test_list_artifact_files_returns_expected_download_entries(tmp_path: Path) -> None:
    """Test list artifact files returns expected download entries.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    run_dir = tmp_path / "run_20260310_101010"
    write_valid_run_dir(run_dir)

    artifacts = list_artifact_files(run_dir)

    assert artifacts[0].label == "Ranked Providers"
    assert artifacts[0].exists
    assert artifacts[0].download_name == "run_20260310_101010_ranked_providers.toml"


def test_list_artifact_files_reports_missing_entries(tmp_path: Path) -> None:
    """Test list artifact files reports missing entries.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    run_dir = tmp_path / "run_missing"
    run_dir.mkdir()

    artifacts = list_artifact_files(run_dir)

    assert all(not artifact.exists for artifact in artifacts)


def test_load_run_summary_handles_missing_files(tmp_path: Path) -> None:
    """Test load run summary handles missing files.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    run_dir = tmp_path / "run_empty"
    run_dir.mkdir()
    summary = load_run_summary(run_dir)
    assert summary.providers == []
    assert summary.objections == []
    assert summary.meeting_scripts == []
    assert summary.retrieval_debug == []
    assert summary.metadata == {}
    assert not summary.validation.is_valid


def test_load_run_bundle_populates_validation_and_artifacts(tmp_path: Path) -> None:
    """Test load run bundle populates validation and artifacts.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    run_dir = tmp_path / "run_bundle"
    write_valid_run_dir(run_dir)

    bundle = load_run_bundle(run_dir)

    assert bundle.validation_report is not None
    assert bundle.validation_report.is_valid
    assert len(bundle.artifacts) == 5
    assert bundle.artifacts[0].exists


def test_summarize_runs_uses_metadata_and_counts(tmp_path: Path) -> None:
    """Test summarize runs uses metadata and counts.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    from tempus_copilot.ui_service import summarize_runs

    run_dir = tmp_path / "run_20260310_222222"
    write_valid_run_dir(run_dir)
    ranked = {
        "schema_version": "1.0.0",
        "providers": [
            {
                "provider_id": "P001",
                "physician_name": "Dr Example",
                "institution": "North",
                "score": 0.5,
                "rationale": "fit",
            }
        ],
    }
    objections = {
        "schema_version": "1.0.0",
        "objections": [
            {
                "provider_id": "P001",
                "concern": "budget",
                "response": "response",
                "supporting_metrics": [],
                "citations": [],
                "confidence": 0.5,
            }
        ],
    }
    scripts = {
        "schema_version": "1.0.0",
        "scripts": [
            {
                "provider_id": "P001",
                "tumor_focus": "Lung",
                "script": "hello",
                "citations": [],
                "confidence": 0.5,
            }
        ],
    }
    metadata = {
        "schema_version": "1.0.0",
        "generated_at_utc": "2026-03-10T00:00:00Z",
        "provider_count": 9,
        "generation_model": "ministral-3:8b",
        "embedding_model": "embeddinggemma",
        "output_checksum_sha256": "bad",
        "baml_schema_sha256": "schema",
        "baml_prompt_sha256": "prompt",
    }
    (run_dir / "ranked_providers.toml").write_bytes(tomli_w.dumps(ranked).encode("utf-8"))
    (run_dir / "objection_handlers.toml").write_bytes(tomli_w.dumps(objections).encode("utf-8"))
    (run_dir / "meeting_scripts.toml").write_bytes(tomli_w.dumps(scripts).encode("utf-8"))
    (run_dir / "run_metadata.toml").write_bytes(tomli_w.dumps(metadata).encode("utf-8"))

    summaries = summarize_runs(tmp_path)

    assert summaries == [
        RunSummary(
            run_dir=run_dir,
            generated_at_utc="2026-03-10T00:00:00Z",
            provider_count=9,
            generation_model="ministral-3:8b",
            embedding_model="embeddinggemma",
            validation_errors=["Checksum mismatch between metadata and outputs"],
            objection_count=1,
            script_count=1,
        )
    ]


def test_private_helpers_cover_coercion_edges() -> None:
    """Test private helpers cover coercion edges.
    """
    from tempus_copilot import ui_service

    assert ui_service._get_mapping("not-a-dict") == {}
    assert ui_service._get_mapping_list({"providers": "bad"}, "providers") == []
    assert ui_service._get_mapping_list(
        {"providers": [{}, {"provider_id": "P1"}]},
        "providers",
    ) == [{"provider_id": "P1"}]
    assert ui_service._get_float({"score": "1.25"}, "score") == 1.25
    assert ui_service._get_float({"score": "bad"}, "score") == 0.0
    assert ui_service._get_float({"score": object()}, "score") == 0.0
    assert ui_service._get_float({"score": 2}, "score") == 2.0
    assert ui_service._coerce_metadata_int(3.0) == 3
    assert ui_service._coerce_metadata_int(3) == 3
    assert ui_service._coerce_metadata_int(3.2) is None
    assert ui_service._get_string_list({"citations": "bad"}, "citations") == []
    assert ui_service._get_string_list({"citations": ["a", 2, "b"]}, "citations") == ["a", "b"]


def test_private_helpers_cover_payload_and_mapping_edges(tmp_path: Path) -> None:
    """Test private helpers cover payload and mapping edges.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    from tempus_copilot import ui_service

    payload_path = tmp_path / "payload.toml"
    payload_path.write_text('schema_version = "1.0.0"\n', encoding="utf-8")

    assert ui_service._load_payload(tmp_path / "missing.toml") == {}
    assert ui_service._load_payload(payload_path) == {"schema_version": "1.0.0"}
    assert ui_service._get_mapping_list(
        {"providers": [0, {}, {"provider_id": "P9"}]},
        "providers",
    ) == [{"provider_id": "P9"}]
    assert ui_service._get_str({"provider_id": 99}, "provider_id") == "99"


def test_load_metadata_handles_optional_and_invalid_types(tmp_path: Path) -> None:
    """Test load metadata handles optional and invalid types.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    from tempus_copilot import ui_service

    run_dir = tmp_path / "run_metadata"
    run_dir.mkdir()
    payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": "2026-03-10T00:00:00Z",
        "provider_count": 2.0,
        "note_count": 3,
        "kb_doc_count": 4.0,
        "kb_chunk_count": 8,
        "generation_model": "ministral-3:8b",
        "embedding_model": "embeddinggemma",
        "output_checksum_sha256": "checksum",
        "baml_schema_sha256": "schema",
        "baml_prompt_sha256": "prompt",
    }
    (run_dir / "run_metadata.toml").write_bytes(tomli_w.dumps(payload).encode("utf-8"))

    metadata = ui_service._load_metadata(run_dir)

    assert metadata["provider_count"] == 2
    assert metadata["note_count"] == 3
    assert metadata["kb_doc_count"] == 4
    assert metadata["kb_chunk_count"] == 8
    assert metadata["generation_model"] == "ministral-3:8b"


def test_load_metadata_ignores_invalid_optional_types(tmp_path: Path) -> None:
    """Test load metadata ignores invalid optional types.
    
    Args:
        tmp_path: Temporary path provided by pytest.
    """
    from tempus_copilot import ui_service

    run_dir = tmp_path / "run_metadata_invalid"
    run_dir.mkdir()
    (run_dir / "run_metadata.toml").write_text(
        "\n".join(
            [
                'schema_version = 1',
                'generated_at_utc = 2',
                'provider_count = 2.5',
                'note_count = "3"',
                'kb_chunk_count = 4.2',
                'generation_model = 9',
                'embedding_model = false',
                'output_checksum_sha256 = 7',
                'baml_schema_sha256 = 8',
                'baml_prompt_sha256 = 9',
            ]
        ),
        encoding="utf-8",
    )

    metadata = ui_service._load_metadata(run_dir)

    assert metadata == {}


def test_dataclass_properties_cover_branchless_helpers() -> None:
    """Test dataclass properties cover branchless helpers.
    """
    artifact = ArtifactFileView(
        file_name="ranked_providers.toml",
        label="Ranked Providers",
        path=Path("outputs/run_1/ranked_providers.toml"),
        exists=True,
    )
    status = ValidationFileStatus(
        file_name="ranked_providers.toml",
        label="Ranked Providers",
        exists=True,
        errors=[],
    )
    report = ValidationReport(
        run_dir=Path("outputs/run_1"),
        file_statuses=[status],
        checksum_error=None,
        errors=[],
    )

    assert artifact.download_name == "run_1_ranked_providers.toml"
    assert status.is_valid
    assert report.is_valid
