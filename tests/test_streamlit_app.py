from __future__ import annotations

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
    bundle = RunBundle(
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

    assert calls == ["Validation passed for `outputs/run_20260310_120000`."]
