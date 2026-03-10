from __future__ import annotations

import importlib
import runpy
from pathlib import Path

import pytest

import streamlit_app
from tempus_copilot.ui_service import (
    MeetingScriptView,
    ObjectionView,
    RankedProviderView,
    RetrievalDebugView,
    RetrievalHitView,
    RunBundle,
    RunMetadataView,
    RunSummary,
    ValidationSummary,
)


class _Context:
    def __enter__(self) -> _Context:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _Sidebar:
    def __init__(self) -> None:
        self.markdown_calls: list[str] = []
        self.caption_calls: list[str] = []

    def markdown(self, value: str) -> None:
        self.markdown_calls.append(value)

    def text_input(self, label: str, value: str) -> str:
        return value

    def radio(self, label: str, options: list[str]) -> str:
        return options[0]

    def caption(self, value: str) -> None:
        self.caption_calls.append(value)


class _AboutSidebar(_Sidebar):
    def radio(self, label: str, options: list[str]) -> str:
        return options[3]


def _sample_bundle() -> RunBundle:
    return RunBundle(
        run_dir=Path("outputs/run_1"),
        ranked_providers=[
            RankedProviderView(
                provider_id="P001",
                physician_name="Dr. A",
                institution="River",
                score=0.9,
                rationale="high fit",
                factor_scores={"patient_volume": 0.7},
                calibration_terms={"specialty_fit": 1.0},
                factor_contributions={"patient_volume": 0.315},
            )
        ],
        objections=[
            ObjectionView(
                provider_id="P001",
                concern="turnaround_time",
                response="Use evidence.",
                supporting_metrics=["8 days"],
                citations=["product_kb.md:0"],
                confidence=0.8,
            )
        ],
        scripts=[
            MeetingScriptView(
                provider_id="P001",
                tumor_focus="Lung",
                script="Lead with clinical fit.",
                citations=["product_kb.md:0"],
                confidence=0.75,
            )
        ],
        retrieval_debug=[
            RetrievalDebugView(
                provider_id="P001",
                query_text="Provider P001.",
                retrieved=[
                    RetrievalHitView(
                        chunk_id="product_kb.md:0",
                        source="product_kb.md",
                        distance=0.1,
                    )
                ],
            )
        ],
        metadata=RunMetadataView(
            schema_version="1.0.0",
            generated_at_utc="2026-03-10T00:00:00Z",
            provider_count=1,
            note_count=1,
            kb_doc_count=1,
            kb_chunk_count=1,
            generation_model="ministral-3:8b",
            embedding_model="embeddinggemma",
            output_checksum_sha256="checksum",
            baml_schema_sha256="schema",
            baml_prompt_sha256="prompt",
        ),
        validation_errors=[],
    )


def test_load_stylesheet_returns_empty_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.css"
    assert streamlit_app.load_stylesheet(missing) == ""


def test_load_stylesheet_reads_existing_file(tmp_path: Path) -> None:
    stylesheet = tmp_path / "streamlit.css"
    stylesheet.write_text(".stApp { color: black; }", encoding="utf-8")
    assert streamlit_app.load_stylesheet(stylesheet) == ".stApp { color: black; }"


def test_format_validation_helpers() -> None:
    assert streamlit_app.format_validation_label(0) == "Validated"
    assert streamlit_app.format_validation_label(2) == "2 issue(s)"
    assert streamlit_app._confidence_threshold(True, 0.65) == 0.65
    assert streamlit_app._confidence_threshold(False, 0.65) is None


def test_format_run_label_reports_validation_state() -> None:
    summary = RunSummary(
        run_dir=Path("outputs/run_20260310_120000"),
        generated_at_utc="2026-03-10T12:00:00+00:00",
        provider_count=3,
        generation_model="ministral-3:8b",
        embedding_model="embeddinggemma",
        validation_errors=[],
    )
    assert "Validated" in streamlit_app._format_run_label(summary)


def test_render_metric_cards_writes_expected_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake_markdown(value: str, *, unsafe_allow_html: bool = False) -> None:
        captured.append(value)
        assert unsafe_allow_html is True

    monkeypatch.setattr(streamlit_app.st, "markdown", fake_markdown)
    bundle = _sample_bundle()
    streamlit_app.render_metric_cards(bundle)
    assert "Providers Ranked" in captured[0]
    assert "Validated" in captured[0]


def test_main_dispatches_selected_page(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeSidebar:
        def markdown(self, value: str) -> None:
            calls.append(f"sidebar-markdown:{value}")

        def text_input(self, label: str, value: str) -> str:
            assert label == "Config Path"
            return value

        def radio(self, label: str, options: list[str]) -> str:
            assert label == "Workspace"
            return options[1]

        def caption(self, value: str) -> None:
            calls.append(f"caption:{value}")

    monkeypatch.setattr(streamlit_app, "load_dotenv", lambda: None)
    monkeypatch.setattr(streamlit_app.st, "set_page_config", lambda **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "sidebar", FakeSidebar())
    monkeypatch.setattr(streamlit_app.st, "session_state", {}, raising=False)
    monkeypatch.setattr(streamlit_app, "load_stylesheet", lambda path=None: "")
    monkeypatch.setattr(
        streamlit_app,
        "load_default_settings",
        lambda path: type("SettingsStub", (), {"output_dir": Path("outputs")})(),
    )
    monkeypatch.setattr(streamlit_app, "summarize_runs", lambda output_dir: [])
    monkeypatch.setattr(streamlit_app, "render_run_page", lambda path: calls.append("run"))
    monkeypatch.setattr(streamlit_app, "render_runs_page", lambda path: calls.append("browse"))
    monkeypatch.setattr(
        streamlit_app,
        "render_validate_page",
        lambda path: calls.append("validate"),
    )
    monkeypatch.setattr(streamlit_app, "render_about_page", lambda path: calls.append("about"))

    streamlit_app.main()

    assert "browse" in calls


def test_main_injects_stylesheet_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeSidebar:
        def markdown(self, value: str) -> None:
            calls.append(value)

        def text_input(self, label: str, value: str) -> str:
            return value

        def radio(self, label: str, options: list[str]) -> str:
            return options[3]

        def caption(self, value: str) -> None:
            calls.append(value)

    captured_markdown: list[str] = []

    def fake_markdown(value: str, *, unsafe_allow_html: bool = False) -> None:
        if unsafe_allow_html:
            captured_markdown.append(value)

    monkeypatch.setattr(streamlit_app, "load_dotenv", lambda: None)
    monkeypatch.setattr(streamlit_app.st, "set_page_config", lambda **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "markdown", fake_markdown)
    monkeypatch.setattr(streamlit_app.st, "caption", lambda value: None)
    monkeypatch.setattr(streamlit_app.st, "sidebar", FakeSidebar())
    monkeypatch.setattr(streamlit_app.st, "session_state", {}, raising=False)
    monkeypatch.setattr(streamlit_app, "load_stylesheet", lambda path=None: ".stApp{}")
    monkeypatch.setattr(
        streamlit_app,
        "render_about_page",
        lambda path: calls.append(f"about:{path}"),
    )

    streamlit_app.main()

    assert captured_markdown[0] == "<style>.stApp{}</style>"


def test_render_validate_page_shows_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class SettingsStub:
        output_dir = Path("outputs")

    monkeypatch.setattr(streamlit_app, "load_default_settings", lambda path: SettingsStub())
    monkeypatch.setattr(streamlit_app, "summarize_runs", lambda output_dir: [])
    monkeypatch.setattr(streamlit_app.st, "selectbox", lambda label, options: "Manual Path")
    monkeypatch.setattr(
        streamlit_app.st,
        "text_input",
        lambda label, value="": "outputs/run_20260310_120000",
    )
    monkeypatch.setattr(
        streamlit_app,
        "validate_run_summary",
        lambda path: ValidationSummary(run_dir=path, errors=[]),
    )
    monkeypatch.setattr(streamlit_app.st, "success", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "info", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "error", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda *args, **kwargs: None)

    streamlit_app.render_validate_page(Path("config/defaults.toml"))

    assert calls == [f"Validation passed for `{Path('outputs/run_20260310_120000')}`."]


def test_validation_cards_include_checksum_state() -> None:
    summary = ValidationSummary(
        run_dir=Path("outputs/run_1"),
        errors=["Checksum mismatch between metadata and outputs"],
    )
    cards = streamlit_app._validation_cards(summary)
    checksum = next(card for card in cards if card.title == "checksum")
    assert checksum.state == "error"


def test_validation_cards_include_file_specific_errors() -> None:
    summary = ValidationSummary(
        run_dir=Path("outputs/run_1"),
        errors=["ranked_providers.toml missing key: providers"],
    )

    cards = streamlit_app._validation_cards(summary)

    provider_card = next(card for card in cards if card.title == "ranked_providers.toml")
    assert provider_card.state == "error"


def test_validation_cards_report_pass_state() -> None:
    summary = ValidationSummary(run_dir=Path("outputs/run_1"), errors=[])
    cards = streamlit_app._validation_cards(summary)
    checksum = next(card for card in cards if card.title == "checksum")
    assert checksum.state == "pass"
    assert len(cards) == 6


def test_metadata_value_covers_string_number_and_fallback() -> None:
    assert streamlit_app._metadata_value({"provider_count": 3}, "provider_count", "x") == "3"
    assert streamlit_app._metadata_value({"schema_version": ""}, "schema_version", "x") == "x"
    assert streamlit_app._metadata_value("bad", "schema_version", "fallback") == "fallback"


def test_filtered_providers_applies_query_and_min_score() -> None:
    bundle = RunBundle(
        run_dir=Path("outputs/run_1"),
        ranked_providers=[
            RankedProviderView(
                provider_id="P001",
                physician_name="Dr. Atlas",
                institution="North",
                score=0.82,
                rationale="lung fit",
                factor_scores={},
                calibration_terms={},
                factor_contributions={},
            ),
            RankedProviderView(
                provider_id="P002",
                physician_name="Dr. Birch",
                institution="South",
                score=0.55,
                rationale="lower volume",
                factor_scores={},
                calibration_terms={},
                factor_contributions={},
            ),
        ],
        objections=[],
        scripts=[],
        retrieval_debug=[],
        metadata=RunMetadataView(),
        validation_errors=[],
    )

    filtered = streamlit_app._filtered_providers(bundle, "atlas", 0.6)

    assert [provider.provider_id for provider in filtered] == ["P001"]
    assert [
        provider.provider_id
        for provider in streamlit_app._filtered_providers(bundle, "", 0.5)
    ] == ["P001", "P002"]


def test_filtered_providers_returns_all_above_threshold_for_empty_query() -> None:
    bundle = RunBundle(
        run_dir=Path("outputs/run_1"),
        ranked_providers=[
            RankedProviderView(
                provider_id="P001",
                physician_name="Dr. Atlas",
                institution="North",
                score=0.82,
                rationale="lung fit",
                factor_scores={},
                calibration_terms={},
                factor_contributions={},
            ),
            RankedProviderView(
                provider_id="P002",
                physician_name="Dr. Birch",
                institution="South",
                score=0.55,
                rationale="lower volume",
                factor_scores={},
                calibration_terms={},
                factor_contributions={},
            ),
        ],
        objections=[],
        scripts=[],
        retrieval_debug=[],
        metadata=RunMetadataView(),
        validation_errors=[],
    )

    filtered = streamlit_app._filtered_providers(bundle, "", 0.6)

    assert [provider.provider_id for provider in filtered] == ["P001"]


def test_metadata_value_uses_fallback_for_unexpected_inputs() -> None:
    assert streamlit_app._metadata_value({}, "schema_version", "fallback") == "fallback"
    assert streamlit_app._metadata_value({"count": 3}, "count", "fallback") == "3"
    assert streamlit_app._metadata_value("bad", "count", "fallback") == "fallback"

def test_render_validate_page_shows_error_list(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class SettingsStub:
        output_dir = Path("outputs")

    monkeypatch.setattr(streamlit_app, "load_default_settings", lambda path: SettingsStub())
    monkeypatch.setattr(streamlit_app, "summarize_runs", lambda output_dir: [])
    monkeypatch.setattr(streamlit_app.st, "selectbox", lambda label, options: "Manual Path")
    monkeypatch.setattr(streamlit_app.st, "text_input", lambda *args, **kwargs: "outputs/run_bad")
    monkeypatch.setattr(streamlit_app, "_render_section_intro", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        streamlit_app,
        "validate_run_summary",
        lambda path: ValidationSummary(run_dir=path, errors=["broken", "missing"]),
    )
    monkeypatch.setattr(streamlit_app.st, "error", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda value, **kwargs: calls.append(value))

    streamlit_app.render_validate_page(Path("config/defaults.toml"))

    assert calls[0] == f"Validation failed for `{Path('outputs/run_bad')}`."
    assert "- broken" in calls[1]
    assert "- missing" in calls[2]


def test_render_about_page_outputs_runtime_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(
        streamlit_app.st,
        "markdown",
        lambda value, **kwargs: captured.append(value),
    )

    streamlit_app.render_about_page(Path("config/defaults.toml"))

    assert "Ollama Cloud only" in captured[-1]


def test_provider_and_retrieval_rows_cover_helpers() -> None:
    bundle = _sample_bundle()
    provider_rows = streamlit_app._provider_rows(bundle)
    retrieval_rows = streamlit_app._retrieval_rows(bundle.retrieval_debug[0])

    assert provider_rows[0]["provider_id"] == "P001"
    assert retrieval_rows[0]["chunk_id"] == "product_kb.md:0"


def test_render_bundle_empty_state_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = RunBundle(
        run_dir=Path("outputs/run_empty"),
        ranked_providers=[],
        objections=[],
        scripts=[],
        retrieval_debug=[],
        metadata=RunMetadataView(),
        validation_errors=["broken"],
    )
    calls: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "dataframe", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "info", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "caption", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "code", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "expander", lambda label: _Context())

    streamlit_app._render_bundle(bundle)

    assert "No ranked providers were written for this run." in calls
    assert "No objection handlers were generated for this run." in calls
    assert "Retrieval diagnostics were not written for this run." in calls


def test_render_bundle_non_empty_retrieval_with_empty_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _sample_bundle()
    bundle = RunBundle(
        run_dir=bundle.run_dir,
        ranked_providers=bundle.ranked_providers,
        objections=bundle.objections,
        scripts=bundle.scripts,
        retrieval_debug=[RetrievalDebugView(provider_id="P001", query_text="", retrieved=[])],
        metadata=bundle.metadata,
        validation_errors=[],
    )
    calls: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda value: calls.append(value))
    monkeypatch.setattr(
        streamlit_app.st,
        "dataframe",
        lambda *args, **kwargs: calls.append("frame"),
    )
    monkeypatch.setattr(streamlit_app.st, "info", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "caption", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "code", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "expander", lambda label: _Context())

    streamlit_app._render_bundle(bundle)

    assert "No retrieval hits were recorded for this provider." in calls


def test_render_bundle_non_empty_retrieval_with_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _sample_bundle()
    calls: list[str] = []
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda value: calls.append(value))
    monkeypatch.setattr(
        streamlit_app.st,
        "dataframe",
        lambda *args, **kwargs: calls.append("frame"),
    )
    monkeypatch.setattr(streamlit_app.st, "info", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "caption", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "code", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "expander", lambda label: _Context())

    streamlit_app._render_bundle(bundle)

    assert "frame" in calls


def test_show_validation_banner_renders_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(
        streamlit_app.st,
        "markdown",
        lambda value, **kwargs: captured.append(value),
    )

    streamlit_app._show_validation_banner(2)

    assert "2 issue(s)" in captured[0]


def test_build_run_overrides_returns_expected_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from tempus_copilot.config import load_settings

    settings = load_settings(Path("config/defaults.toml"))

    values = iter(
        [
            "market.csv",
            "kb.md",
            "gen",
            "crm.csv",
            "outputs",
            "embed",
        ]
    )
    numbers = iter([100, 4, 10, 2, 0.5])
    checkboxes = iter([True, True])

    monkeypatch.setattr(streamlit_app.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "form", lambda *args, **kwargs: _Context())
    monkeypatch.setattr(
        streamlit_app.st,
        "columns",
        lambda count: [_Context() for _ in range(count)],
    )
    monkeypatch.setattr(streamlit_app.st, "text_input", lambda *args, **kwargs: next(values))
    monkeypatch.setattr(
        streamlit_app.st,
        "number_input",
        lambda *args, **kwargs: next(numbers),
    )
    monkeypatch.setattr(
        streamlit_app.st,
        "checkbox",
        lambda *args, **kwargs: next(checkboxes),
    )
    monkeypatch.setattr(streamlit_app.st, "slider", lambda *args, **kwargs: 0.7)
    monkeypatch.setattr(streamlit_app.st, "form_submit_button", lambda *args, **kwargs: True)

    overrides, controls, submitted = streamlit_app._build_run_overrides(settings)

    assert overrides.strict_citations is True
    assert controls.fail_on_low_confidence == 0.7
    assert submitted is True


def test_render_run_page_covers_success_failure_and_empty_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = type("SettingsStub", (), {"output_dir": Path("outputs")})()
    bundle = _sample_bundle()
    calls: list[str] = []

    monkeypatch.setattr(streamlit_app, "load_default_settings", lambda path: settings)
    monkeypatch.setattr(streamlit_app, "summarize_runs", lambda output_dir: [])
    monkeypatch.setattr(
        streamlit_app,
        "_build_run_overrides",
        lambda settings: (streamlit_app.SettingsOverride(), streamlit_app.RunControls(), False),
    )
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "caption", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "session_state", {}, raising=False)
    monkeypatch.setattr(streamlit_app, "load_run_bundle", lambda path: bundle)

    streamlit_app.render_run_page(Path("config/defaults.toml"))
    assert calls[-1] == "Run the pipeline to render ranked output and validation details here."

    monkeypatch.setattr(
        streamlit_app,
        "_build_run_overrides",
        lambda settings: (streamlit_app.SettingsOverride(), streamlit_app.RunControls(), True),
    )
    monkeypatch.setattr(streamlit_app.st, "spinner", lambda message: _Context())
    monkeypatch.setattr(
        streamlit_app,
        "run_pipeline_from_ui",
        lambda *args, **kwargs: Path("outputs/run_success"),
    )
    monkeypatch.setattr(streamlit_app.st, "success", lambda value: calls.append(value))
    monkeypatch.setattr(
        streamlit_app,
        "_render_bundle",
        lambda value: calls.append("bundle"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "_show_validation_banner",
        lambda value: calls.append(f"banner:{value}"),
    )

    streamlit_app.render_run_page(Path("config/defaults.toml"))
    assert f"Run completed: {Path('outputs/run_success')}" in calls
    assert "bundle" in calls

    monkeypatch.setattr(
        streamlit_app,
        "run_pipeline_from_ui",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(streamlit_app.st, "error", lambda value: calls.append(value))
    streamlit_app.render_run_page(Path("config/defaults.toml"))
    assert "Run failed: boom" in calls


def test_render_runs_page_empty_and_populated(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = type("SettingsStub", (), {"output_dir": Path("outputs")})()
    bundle = _sample_bundle()
    summary = RunSummary(
        run_dir=bundle.run_dir,
        generated_at_utc="2026-03-10T00:00:00Z",
        provider_count=1,
        generation_model="ministral-3:8b",
        embedding_model="embeddinggemma",
        validation_errors=[],
    )
    calls: list[str] = []

    monkeypatch.setattr(streamlit_app, "load_default_settings", lambda path: settings)
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "warning", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app, "summarize_runs", lambda output_dir: [])

    streamlit_app.render_runs_page(Path("config/defaults.toml"))
    assert calls == ["No run directories were found in the configured output directory."]

    monkeypatch.setattr(streamlit_app, "summarize_runs", lambda output_dir: [summary])
    monkeypatch.setattr(
        streamlit_app.st,
        "selectbox",
        lambda label, options: options[0],
    )
    monkeypatch.setattr(streamlit_app, "load_run_bundle", lambda run_dir: bundle)
    monkeypatch.setattr(
        streamlit_app,
        "_show_validation_banner",
        lambda value: calls.append(f"banner:{value}"),
    )
    monkeypatch.setattr(streamlit_app, "_render_bundle", lambda value: calls.append("bundle"))

    streamlit_app.render_runs_page(Path("config/defaults.toml"))

    assert "banner:0" in calls
    assert "bundle" in calls


def test_render_validate_page_handles_selected_run_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = type("SettingsStub", (), {"output_dir": Path("outputs")})()
    summary = RunSummary(
        run_dir=Path("outputs/run_selected"),
        generated_at_utc="2026-03-10T00:00:00Z",
        provider_count=1,
        generation_model="ministral-3:8b",
        embedding_model="embeddinggemma",
        validation_errors=["broken"],
    )
    calls: list[str] = []

    monkeypatch.setattr(streamlit_app, "load_default_settings", lambda path: settings)
    monkeypatch.setattr(streamlit_app, "summarize_runs", lambda output_dir: [summary])
    monkeypatch.setattr(
        streamlit_app.st,
        "selectbox",
        lambda label, options: options[1],
    )
    monkeypatch.setattr(streamlit_app.st, "text_input", lambda label, value="": value)
    monkeypatch.setattr(
        streamlit_app,
        "validate_run_summary",
        lambda path: ValidationSummary(run_dir=path, errors=["missing key"]),
    )
    monkeypatch.setattr(streamlit_app.st, "success", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "error", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "info", lambda value: calls.append(value))
    monkeypatch.setattr(
        streamlit_app.st,
        "markdown",
        lambda value, **kwargs: calls.append(value) if str(value).startswith("-") else None,
    )

    streamlit_app.render_validate_page(Path("config/defaults.toml"))

    assert f"Validation failed for `{Path('outputs/run_selected')}`." in calls


def test_render_validate_page_handles_blank_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = type("SettingsStub", (), {"output_dir": Path("outputs")})()
    calls: list[str] = []
    monkeypatch.setattr(streamlit_app, "load_default_settings", lambda path: settings)
    monkeypatch.setattr(streamlit_app, "summarize_runs", lambda output_dir: [])
    monkeypatch.setattr(streamlit_app.st, "selectbox", lambda label, options: "Manual Path")
    monkeypatch.setattr(streamlit_app.st, "text_input", lambda label, value="": "   ")
    monkeypatch.setattr(streamlit_app.st, "info", lambda value: calls.append(value))
    monkeypatch.setattr(streamlit_app.st, "markdown", lambda *args, **kwargs: None)

    streamlit_app.render_validate_page(Path("config/defaults.toml"))

    assert calls == ["Select an existing run or enter a run directory to validate."]


def test_render_about_page_and_main_default_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    sidebar = _AboutSidebar()
    calls: list[str] = []

    def fake_markdown(value: str, **kwargs: object) -> None:
        calls.append(value)

    monkeypatch.setattr(streamlit_app, "load_dotenv", lambda: None)
    monkeypatch.setattr(streamlit_app.st, "set_page_config", lambda **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "markdown", fake_markdown)
    monkeypatch.setattr(streamlit_app.st, "caption", lambda value: None)
    monkeypatch.setattr(streamlit_app.st, "sidebar", sidebar)
    monkeypatch.setattr(streamlit_app.st, "session_state", {}, raising=False)
    monkeypatch.setattr(streamlit_app, "load_stylesheet", lambda path=None: "")
    monkeypatch.setattr(
        streamlit_app,
        "load_default_settings",
        lambda path: type("SettingsStub", (), {"output_dir": Path("outputs")})(),
    )
    monkeypatch.setattr(streamlit_app, "summarize_runs", lambda output_dir: [])
    monkeypatch.setattr(streamlit_app, "render_run_page", lambda path: calls.append("run"))
    monkeypatch.setattr(streamlit_app, "render_runs_page", lambda path: calls.append("browse"))
    monkeypatch.setattr(
        streamlit_app,
        "render_validate_page",
        lambda path: calls.append("validate"),
    )
    monkeypatch.setattr(streamlit_app, "render_about_page", lambda path: calls.append("about"))

    streamlit_app.main()
    streamlit_app.render_about_page(Path("config/defaults.toml"))

    assert "about" in calls


def test_main_run_and_validate_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    for index, expected in [(0, "run"), (2, "validate")]:
        sidebar = _Sidebar()
        calls: list[str] = []
        monkeypatch.setattr(streamlit_app, "load_dotenv", lambda: None)
        monkeypatch.setattr(streamlit_app.st, "set_page_config", lambda **kwargs: None)
        monkeypatch.setattr(streamlit_app.st, "markdown", lambda *args, **kwargs: None)
        monkeypatch.setattr(streamlit_app.st, "caption", lambda value: None)
        monkeypatch.setattr(streamlit_app.st, "sidebar", sidebar)
        monkeypatch.setattr(streamlit_app.st, "session_state", {}, raising=False)
        monkeypatch.setattr(streamlit_app, "load_stylesheet", lambda path=None: "")
        monkeypatch.setattr(
            streamlit_app,
            "load_default_settings",
            lambda path: type("SettingsStub", (), {"output_dir": Path("outputs")})(),
        )
        monkeypatch.setattr(streamlit_app, "summarize_runs", lambda output_dir: [])
        monkeypatch.setattr(
            streamlit_app,
            "render_run_page",
            lambda path, calls=calls: calls.append("run"),
        )
        monkeypatch.setattr(
            streamlit_app,
            "render_runs_page",
            lambda path, calls=calls: calls.append("browse"),
        )
        monkeypatch.setattr(
            streamlit_app,
            "render_validate_page",
            lambda path, calls=calls: calls.append("validate"),
        )
        monkeypatch.setattr(
            streamlit_app,
            "render_about_page",
            lambda path, calls=calls: calls.append("about"),
        )
        sidebar.radio = lambda label, options, idx=index: options[idx]  # type: ignore[method-assign]

        streamlit_app.main()

        assert expected in calls


def test_module_reload_adds_src_to_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    src_dir = str(Path(streamlit_app.__file__).resolve().parent / "src")
    original = list(streamlit_app.sys.path)
    monkeypatch.setattr(streamlit_app.sys, "path", [path for path in original if path != src_dir])

    reloaded = importlib.reload(streamlit_app)

    assert src_dir in reloaded.sys.path


def test_main_guard_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    sidebar = _AboutSidebar()

    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setattr("streamlit.set_page_config", lambda **kwargs: None)
    monkeypatch.setattr("streamlit.markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr("streamlit.caption", lambda value: None)
    monkeypatch.setattr("streamlit.sidebar", sidebar)
    monkeypatch.setattr("streamlit.session_state", {}, raising=False)
    monkeypatch.setattr(
        "tempus_copilot.ui_service.load_default_settings",
        lambda path: type("SettingsStub", (), {"output_dir": Path("outputs")})(),
    )
    monkeypatch.setattr("tempus_copilot.ui_service.summarize_runs", lambda output_dir: [])

    namespace = runpy.run_path(str(Path("streamlit_app.py")), run_name="__main__")

    assert namespace["__name__"] == "__main__"
