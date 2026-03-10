"""Streamlit operator workspace for the TempoTalk pipeline."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tempus_copilot.config import Settings  # noqa: E402
from tempus_copilot.ui_service import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    MeetingScriptView,
    ObjectionView,
    RankedProviderView,
    RetrievalDebugView,
    RunBundle,
    RunControls,
    RunSummary,
    SettingsOverride,
    ValidationSummary,
    load_default_settings,
    load_run_bundle,
    run_pipeline_from_ui,
    summarize_runs,
    validate_run_summary,
)

APP_TITLE = "TempoTalk Control Room"
APP_SUBTITLE = "Field intelligence for ranked outreach, objections, and meeting prep."
DEFAULT_STYLESHEET = Path("assets/streamlit.css")
SESSION_LAST_RUN = "last_run_dir"
type TableValue = str | float
type ValidationState = str

ARTIFACT_FILES = (
    "ranked_providers.toml",
    "objection_handlers.toml",
    "meeting_scripts.toml",
    "retrieval_debug.toml",
    "run_metadata.toml",
)


@dataclass(frozen=True)
class ValidationCard:
    """Represents a validation status card rendered in the UI."""

    title: str
    state: ValidationState
    detail: str


def load_stylesheet(path: Path | None = None) -> str:
    """Loads the optional Streamlit stylesheet.

    Args:
        path: Override path for the stylesheet file.

    Returns:
        The stylesheet text if the file exists, otherwise an empty string.
    """
    stylesheet = DEFAULT_STYLESHEET if path is None else path
    if not stylesheet.exists():
        return ""
    return stylesheet.read_text(encoding="utf-8")


def format_validation_label(issue_count: int) -> str:
    """Formats validation error counts for badges and banners.

    Args:
        issue_count: Number of validation issues for the run.

    Returns:
        A short label describing validation state.
    """
    if issue_count == 0:
        return "Validated"
    return f"{issue_count} issue(s)"


def _confidence_threshold(enabled: bool, value: float) -> float | None:
    """Converts the confidence toggle into an optional threshold.

    Args:
        enabled: Whether threshold enforcement is enabled.
        value: The selected confidence floor.

    Returns:
        The threshold when enabled, otherwise ``None``.
    """
    if enabled:
        return value
    return None


def _format_run_label(summary: RunSummary) -> str:
    """Builds the run selector label.

    Args:
        summary: Summary information for a run directory.

    Returns:
        A human-readable run label.
    """
    validation_label = format_validation_label(len(summary.validation_errors))
    generated = summary.generated_at_utc or "timestamp unavailable"
    return (
        f"{summary.run_dir.name} | {summary.provider_count} providers | "
        f"{generated} | {validation_label}"
    )


def _render_hero() -> None:
    """Renders the top-of-page hero section."""
    st.markdown(
        """
        <section class="hero-shell">
          <div class="hero-kicker">TempoTalk</div>
          <h1>Control Room</h1>
          <p>
            Launch new runs, inspect ranked providers, and validate every
            schema-versioned artifact from a single operator workspace.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_section_intro(title: str, copy: str) -> None:
    """Renders a shared section intro block.

    Args:
        title: Section label.
        copy: Supporting body copy.
    """
    st.markdown(
        f"""
        <div class="section-intro">
          <span class="section-label">{title}</span>
          <p>{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metadata_value(metadata: object, key: str, fallback: str) -> str:
    """Extracts a displayable metadata value.

    Args:
        metadata: Run metadata payload.
        key: Metadata key to read.
        fallback: Value to use when the key is absent or invalid.

    Returns:
        A normalized string value for display.
    """
    if isinstance(metadata, dict):
        metadata_map = cast(dict[str, object], metadata)
        value = metadata_map.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, int | float):
            return str(value)
    return fallback


def _validation_cards(summary: ValidationSummary) -> list[ValidationCard]:
    """Builds validation cards from a summary object.

    Args:
        summary: Validation result for a run.

    Returns:
        The ordered validation cards shown in the UI.
    """
    cards: list[ValidationCard] = []
    # Keep the card list stable so missing files do not shift the visual layout between runs.
    for file_name in ARTIFACT_FILES:
        errors = [error for error in summary.errors if file_name in error]
        if errors:
            cards.append(
                ValidationCard(
                    title=file_name,
                    state="error",
                    detail=" | ".join(errors),
                )
            )
        else:
            cards.append(
                ValidationCard(
                    title=file_name,
                    state="pass",
                    detail="Required keys present.",
                )
            )
    checksum_errors = [error for error in summary.errors if "Checksum" in error]
    cards.append(
        ValidationCard(
            title="checksum",
            state="error" if checksum_errors else "pass",
            detail=checksum_errors[0] if checksum_errors else "Metadata checksum matches outputs.",
        )
    )
    return cards


def render_metric_cards(bundle: RunBundle) -> None:
    """Renders top-line metrics for a run bundle.

    Args:
        bundle: Loaded run bundle to summarize.
    """
    validation_label = format_validation_label(len(bundle.validation_errors))
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card">
            <span>Providers Ranked</span>
            <strong>{len(bundle.ranked_providers)}</strong>
          </div>
          <div class="metric-card">
            <span>Objection Drafts</span>
            <strong>{len(bundle.objections)}</strong>
          </div>
          <div class="metric-card">
            <span>Meeting Scripts</span>
            <strong>{len(bundle.scripts)}</strong>
          </div>
          <div class="metric-card">
            <span>Validation</span>
            <strong>{validation_label}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metadata(bundle: RunBundle) -> None:
    """Renders key run metadata values.

    Args:
        bundle: Loaded run bundle to summarize.
    """
    metadata = bundle.metadata
    schema = _metadata_value(metadata, "schema_version", "unknown")
    generated = _metadata_value(metadata, "generated_at_utc", "not recorded")
    generation_model = _metadata_value(metadata, "generation_model", "not recorded")
    embedding_model = _metadata_value(metadata, "embedding_model", "not recorded")
    st.markdown(
        f"""
        <div class="meta-grid">
          <div class="meta-card">
            <span>Schema</span>
            <strong>{schema}</strong>
          </div>
          <div class="meta-card">
            <span>Generated</span>
            <strong>{generated}</strong>
          </div>
          <div class="meta-card">
            <span>Generation Model</span>
            <strong>{generation_model}</strong>
          </div>
          <div class="meta-card">
            <span>Embedding Model</span>
            <strong>{embedding_model}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _provider_rows(bundle: RunBundle) -> list[dict[str, TableValue]]:
    """Converts ranked providers into table rows.

    Args:
        bundle: Loaded run bundle.

    Returns:
        Flattened provider rows for tabular display.
    """
    return [
        {
            "provider_id": provider.provider_id,
            "physician_name": provider.physician_name,
            "institution": provider.institution,
            "score": provider.score,
            "rationale": provider.rationale,
        }
        for provider in bundle.ranked_providers
    ]


def _filtered_providers(
    bundle: RunBundle,
    query: str,
    min_score: float,
) -> list[RankedProviderView]:
    """Filters providers by free-text query and minimum score.

    Args:
        bundle: Loaded run bundle.
        query: User-entered search text.
        min_score: Minimum score threshold.

    Returns:
        Providers matching the requested filters.
    """
    providers = [
        provider for provider in bundle.ranked_providers if provider.score >= min_score
    ]
    needle = query.strip().lower()
    if not needle:
        return providers
    return [
        provider
        for provider in providers
        if needle in provider.provider_id.lower()
        or needle in provider.physician_name.lower()
        or needle in provider.institution.lower()
        or needle in provider.rationale.lower()
    ]


def _retrieval_rows(item: RetrievalDebugView) -> list[dict[str, TableValue]]:
    """Converts retrieval hits into table rows.

    Args:
        item: Retrieval debug entry for one provider.

    Returns:
        Flattened retrieval rows for display.
    """
    return [
        {
            "chunk_id": hit.chunk_id,
            "source": hit.source,
            "distance": hit.distance,
        }
        for hit in item.retrieved
    ]


def _render_objection_card(item: ObjectionView) -> None:
    """Renders an objection handler card.

    Args:
        item: Objection artifact to display.
    """
    metrics = ", ".join(item.supporting_metrics) or "-"
    citations = ", ".join(item.citations) or "-"
    confidence = f"{item.confidence:.2f}"
    st.markdown(
        f"""
        <article class="artifact-card">
          <div class="artifact-head">
            <span class="artifact-pill">{item.provider_id}</span>
            <span class="artifact-pill">{item.concern}</span>
            <span class="artifact-pill">confidence {confidence}</span>
          </div>
          <p>{item.response}</p>
          <div class="artifact-foot">metrics: {metrics}</div>
          <div class="artifact-foot">citations: {citations}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def _render_script_card(item: MeetingScriptView) -> None:
    """Renders a meeting script card.

    Args:
        item: Meeting script artifact to display.
    """
    citations = ", ".join(item.citations) or "-"
    confidence = f"{item.confidence:.2f}"
    st.markdown(
        f"""
        <article class="artifact-card">
          <div class="artifact-head">
            <span class="artifact-pill">{item.provider_id}</span>
            <span class="artifact-pill">{item.tumor_focus}</span>
            <span class="artifact-pill">confidence {confidence}</span>
          </div>
          <p>{item.script}</p>
          <div class="artifact-foot">citations: {citations}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def _render_provider_detail(provider: RankedProviderView) -> None:
    """Renders provider rationale details.

    Args:
        provider: Ranked provider entry to display.
    """
    st.markdown(
        f"""
        <article class="artifact-card">
          <div class="artifact-head">
            <span class="artifact-pill">{provider.provider_id}</span>
            <span class="artifact-pill">{provider.physician_name}</span>
            <span class="artifact-pill">{provider.institution}</span>
          </div>
          <p>{provider.rationale}</p>
          <div class="artifact-foot">score: {provider.score:.2f}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def _render_bundle(bundle: RunBundle) -> None:
    """Renders the full bundle detail view.

    Args:
        bundle: Loaded run bundle to display.
    """
    # Render the same canonical bundle shape on the run and browse pages.
    render_metric_cards(bundle)
    _render_metadata(bundle)

    st.subheader("Ranked Providers")
    if bundle.ranked_providers:
        st.dataframe(_provider_rows(bundle), use_container_width=True, hide_index=True)
    else:
        st.info("No ranked providers were written for this run.")

    st.subheader("Provider Notes")
    if bundle.ranked_providers:
        for provider in bundle.ranked_providers:
            _render_provider_detail(provider)
    else:
        st.caption("Provider rationale appears here when ranked output is present.")

    st.subheader("Objection Handlers")
    if bundle.objections:
        for item in bundle.objections:
            _render_objection_card(item)
    else:
        st.caption("No objection handlers were generated for this run.")

    st.subheader("Meeting Scripts")
    if bundle.scripts:
        for item in bundle.scripts:
            _render_script_card(item)
    else:
        st.caption("No meeting scripts were generated for this run.")

    st.subheader("Retrieval Debug")
    if bundle.retrieval_debug:
        for item in bundle.retrieval_debug:
            label = f"{item.provider_id}: {item.query_text or 'query unavailable'}"
            with st.expander(label):
                retrieval_rows = _retrieval_rows(item)
                if retrieval_rows:
                    st.dataframe(retrieval_rows, use_container_width=True, hide_index=True)
                else:
                    st.caption("No retrieval hits were recorded for this provider.")
    else:
        st.caption("Retrieval diagnostics were not written for this run.")

    st.subheader("Run Directory")
    st.code(str(bundle.run_dir), language="text")


def _build_run_overrides(settings: Settings) -> tuple[SettingsOverride, RunControls, bool]:
    """Builds transient run settings from the form.

    Args:
        settings: Baseline settings loaded from config.

    Returns:
        A tuple of settings overrides, run controls, and submit state.
    """
    # The form edits a transient override object; it never rewrites config/defaults.toml.
    st.markdown('<div class="control-block">Run Configuration</div>', unsafe_allow_html=True)
    with st.form("run-pipeline-form", clear_on_submit=False):
        path_left, path_right = st.columns(2)
        with path_left:
            market_csv = st.text_input("Market CSV", value=str(settings.market_csv))
            kb_markdown = st.text_input("Knowledge Base Markdown", value=str(settings.kb_markdown))
            generation_model = st.text_input(
                "Generation Model",
                value=settings.models.generation_model,
            )
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=1,
                value=settings.rag.chunk_size,
                step=32,
            )
            top_k = st.number_input("Top K", min_value=1, value=settings.rag.top_k, step=1)
        with path_right:
            crm_csv = st.text_input("CRM CSV", value=str(settings.crm_csv))
            output_dir = st.text_input("Output Directory", value=str(settings.output_dir))
            embedding_model = st.text_input(
                "Embedding Model",
                value=settings.models.embedding_model,
            )
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                value=settings.rag.chunk_overlap,
                step=8,
            )
            request_retries = st.number_input(
                "Request Retries",
                min_value=0,
                value=settings.rag.request_retries,
                step=1,
            )

        policy_left, policy_right = st.columns(2)
        with policy_left:
            backoff_seconds = st.number_input(
                "Retry Backoff Seconds",
                min_value=0.0,
                value=float(settings.rag.backoff_seconds),
                step=0.25,
            )
            strict_citations = st.checkbox(
                "Strict Citations",
                value=settings.output.strict_citations,
            )
        with policy_right:
            enforce_low_confidence = st.checkbox("Fail on Low Confidence", value=False)
            confidence_floor = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                disabled=not enforce_low_confidence,
            )

        submitted = st.form_submit_button("Run Pipeline", use_container_width=True)

    overrides = SettingsOverride(
        market_csv=Path(market_csv),
        crm_csv=Path(crm_csv),
        kb_markdown=Path(kb_markdown),
        output_dir=Path(output_dir),
        generation_model=generation_model.strip(),
        embedding_model=embedding_model.strip(),
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        top_k=int(top_k),
        request_retries=int(request_retries),
        backoff_seconds=float(backoff_seconds),
        strict_citations=strict_citations,
    )
    controls = RunControls(
        strict_citations=strict_citations,
        fail_on_low_confidence=_confidence_threshold(
            enforce_low_confidence,
            float(confidence_floor),
        ),
    )
    return overrides, controls, submitted


def _show_validation_banner(error_count: int) -> None:
    """Renders the validation status banner.

    Args:
        error_count: Number of validation errors.
    """
    label = format_validation_label(error_count)
    st.markdown(
        f"""
        <div class="insight-banner">
          <strong>Validation status</strong>
          <span>{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_run_page(config_path: Path) -> None:
    """Renders the pipeline execution page.

    Args:
        config_path: Path to the settings file to load.
    """
    _render_section_intro(
        "Run Pipeline",
        "Launch a new pipeline run with in-memory overrides while leaving the checked-in "
        "config untouched.",
    )
    settings = load_default_settings(config_path)
    st.markdown(
        f"""
        <div class="insight-banner">
          <strong>Config source</strong>
          <span>{config_path}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    overrides, controls, submitted = _build_run_overrides(settings)

    if submitted:
        try:
            with st.spinner("Running ranking, retrieval, and generation..."):
                run_dir = run_pipeline_from_ui(
                    config_path,
                    settings_overrides=overrides,
                    controls=controls,
                )
            st.session_state[SESSION_LAST_RUN] = str(run_dir)
            st.success(f"Run completed: {run_dir}")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(f"Run failed: {exc}")
            return

    last_run_value = st.session_state.get(SESSION_LAST_RUN)
    if not isinstance(last_run_value, str) or not last_run_value:
        st.caption("Run the pipeline to render ranked output and validation details here.")
        return

    bundle = load_run_bundle(Path(last_run_value))
    _show_validation_banner(len(bundle.validation_errors))
    _render_bundle(bundle)


def render_runs_page(config_path: Path) -> None:
    """Renders the historical runs browser.

    Args:
        config_path: Path to the settings file to load.
    """
    _render_section_intro(
        "Browse Runs",
        "Audit ranked providers, objection handling, scripts, and retrieval traces from "
        "previous pipeline runs.",
    )
    settings = load_default_settings(config_path)
    summaries = summarize_runs(settings.output_dir)
    if not summaries:
        st.warning("No run directories were found in the configured output directory.")
        return

    options = {_format_run_label(summary): summary for summary in summaries}
    selected_label = st.selectbox("Select Run", list(options.keys()))
    selected = options[selected_label]
    bundle = load_run_bundle(selected.run_dir)

    _show_validation_banner(len(bundle.validation_errors))
    _render_bundle(bundle)


def render_validate_page(config_path: Path) -> None:
    """Renders the output validation page.

    Args:
        config_path: Path to the settings file to load.
    """
    _render_section_intro(
        "Validate Outputs",
        "Check schema keys and checksum integrity before using a run as a reference artifact.",
    )
    settings = load_default_settings(config_path)
    summaries = summarize_runs(settings.output_dir)
    options = ["Manual Path"] + [_format_run_label(summary) for summary in summaries]
    selected_label = st.selectbox("Run Source", options)

    default_path = ""
    if selected_label != "Manual Path":
        selected_summary = next(
            summary for summary in summaries if _format_run_label(summary) == selected_label
        )
        default_path = str(selected_summary.run_dir)

    candidate = st.text_input("Run Directory", value=default_path)
    if not candidate.strip():
        st.info("Select an existing run or enter a run directory to validate.")
        return

    summary = validate_run_summary(Path(candidate))
    if summary.is_valid:
        st.success(f"Validation passed for `{Path(candidate)}`.")
        return

    st.error(f"Validation failed for `{Path(candidate)}`.")
    for error in summary.errors:
        st.markdown(f"- {error}")


def render_about_page(config_path: Path) -> None:
    """Renders runtime and artifact notes for the app.

    Args:
        config_path: Active settings path shown to the operator.
    """
    _render_section_intro(
        "System / About",
        "Reference notes for runtime boundaries, artifacts, and how the Streamlit shell "
        "relates to the CLI.",
    )
    st.markdown(
        f"""
        <article class="artifact-card">
          <div class="artifact-head">
            <span class="artifact-pill">config</span>
            <span class="artifact-pill">{config_path}</span>
          </div>
          <p>
            The frontend calls the typed backend directly. It applies overrides in memory,
            does not mutate <code>config/defaults.toml</code>, and reads the same TOML
            artifacts validated by the CLI.
          </p>
          <div class="artifact-foot">Generation: Ollama Cloud only</div>
          <div class="artifact-foot">Embeddings: local Ollama only</div>
          <div class="artifact-foot">Runtime path: handwritten Ollama adapter</div>
          <div class="artifact-foot">Artifacts: schema-versioned TOML in outputs/run_*</div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Runs the Streamlit app."""
    load_dotenv()
    # Import path bootstrapping happens above so repo-root Streamlit launches
    # can still import the package.
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":material/analytics:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    stylesheet = load_stylesheet()
    if stylesheet:
        st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)

    _render_hero()
    st.caption(APP_SUBTITLE)

    st.sidebar.markdown("### TempoTalk")
    st.sidebar.caption("Field intelligence workspace")
    config_path_text = st.sidebar.text_input("Config Path", value=str(DEFAULT_CONFIG_PATH))
    config_path = Path(config_path_text)
    workspace = st.sidebar.radio(
        "Workspace",
        options=["Run Pipeline", "Browse Runs", "Validate Outputs", "System / About"],
    )

    if workspace == "Run Pipeline":
        render_run_page(config_path)
    elif workspace == "Browse Runs":
        render_runs_page(config_path)
    elif workspace == "Validate Outputs":
        render_validate_page(config_path)
    else:
        render_about_page(config_path)


if __name__ == "__main__":
    main()
