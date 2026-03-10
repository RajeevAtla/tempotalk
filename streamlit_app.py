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
    title: str
    state: ValidationState
    detail: str


@dataclass(frozen=True)
class ArtifactDownload:
    label: str
    path: Path


def load_stylesheet(path: Path | None = None) -> str:
    stylesheet = DEFAULT_STYLESHEET if path is None else path
    if not stylesheet.exists():
        return ""
    return stylesheet.read_text(encoding="utf-8")


def format_validation_label(issue_count: int) -> str:
    if issue_count == 0:
        return "Validated"
    return f"{issue_count} issue(s)"


def _confidence_threshold(enabled: bool, value: float) -> float | None:
    if enabled:
        return value
    return None


def _format_run_label(summary: RunSummary) -> str:
    validation_label = format_validation_label(len(summary.validation_errors))
    generated = summary.generated_at_utc or "timestamp unavailable"
    return (
        f"{summary.run_dir.name} | {summary.provider_count} providers | "
        f"{generated} | {validation_label}"
    )


def _sidebar_markdown(value: str, *, unsafe_allow_html: bool = False) -> None:
    try:
        st.sidebar.markdown(value, unsafe_allow_html=unsafe_allow_html)
    except TypeError:
        st.sidebar.markdown(value)


def _metadata_value(metadata: object, key: str, fallback: str) -> str:
    if isinstance(metadata, dict):
        metadata_map = cast(dict[str, object], metadata)
        value = metadata_map.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, int | float):
            return str(value)
    return fallback


def _metadata_float(metadata: object, key: str) -> float | None:
    if isinstance(metadata, dict):
        metadata_map = cast(dict[str, object], metadata)
        value = metadata_map.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _remember_last_run(run_dir: Path) -> None:
    st.session_state[SESSION_LAST_RUN] = str(run_dir)


def _last_run_dir(summaries: list[RunSummary]) -> Path | None:
    session_value = st.session_state.get(SESSION_LAST_RUN)
    if isinstance(session_value, str) and session_value:
        return Path(session_value)
    if summaries:
        return summaries[0].run_dir
    return None


def _load_latest_bundle(summaries: list[RunSummary]) -> RunBundle | None:
    run_dir = _last_run_dir(summaries)
    if run_dir is None:
        return None
    return load_run_bundle(run_dir)


def _render_hero() -> None:
    st.markdown(
        """
        <section class="hero-shell">
          <div class="hero-kicker">TempoTalk</div>
          <h1>Control Room</h1>
          <p>
            An editorial command desk for ranking providers, pressure-testing objections,
            and carrying validated meeting scripts into the field.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_section_intro(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-intro">
          <span class="section-label">{title}</span>
          <p>{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_status(config_path: Path, summaries: list[RunSummary]) -> None:
    last_run = _last_run_dir(summaries)
    last_label = "No runs yet"
    if last_run is not None:
        last_label = last_run.name
    _sidebar_markdown("### TempoTalk")
    st.sidebar.caption("Field intelligence workspace")
    _sidebar_markdown(
        f"""
        <div class="sidebar-status">
          <span>Config</span>
          <strong>{config_path}</strong>
          <span>Indexed runs</span>
          <strong>{len(summaries)}</strong>
          <span>Last run</span>
          <strong>{last_label}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(bundle: RunBundle) -> None:
    validation_label = format_validation_label(len(bundle.validation_errors))
    top_score = max((provider.score for provider in bundle.ranked_providers), default=0.0)
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
            <span>Top Score</span>
            <strong>{top_score:.2f}</strong>
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
    metadata = bundle.metadata
    schema = _metadata_value(metadata, "schema_version", "unknown")
    generated = _metadata_value(metadata, "generated_at_utc", "not recorded")
    generation_model = _metadata_value(metadata, "generation_model", "not recorded")
    embedding_model = _metadata_value(metadata, "embedding_model", "not recorded")
    checksum = _metadata_value(metadata, "output_checksum_sha256", "not recorded")
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
          <div class="meta-card">
            <span>Checksum</span>
            <strong>{checksum}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _provider_rows(
    providers: list[RankedProviderView],
) -> list[dict[str, TableValue]]:
    return [
        {
            "provider_id": provider.provider_id,
            "physician_name": provider.physician_name,
            "institution": provider.institution,
            "score": provider.score,
            "rationale": provider.rationale,
        }
        for provider in providers
    ]


def _retrieval_rows(item: RetrievalDebugView) -> list[dict[str, TableValue]]:
    return [
        {
            "chunk_id": hit.chunk_id,
            "source": hit.source,
            "distance": hit.distance,
        }
        for hit in item.retrieved
    ]


def _artifact_downloads(run_dir: Path) -> list[ArtifactDownload]:
    downloads: list[ArtifactDownload] = []
    for file_name in ARTIFACT_FILES:
        artifact_path = run_dir / file_name
        if artifact_path.exists():
            downloads.append(ArtifactDownload(label=file_name, path=artifact_path))
    return downloads


def _validation_cards(summary: ValidationSummary) -> list[ValidationCard]:
    cards: list[ValidationCard] = []
    for file_name in ARTIFACT_FILES:
        relevant = [error for error in summary.errors if file_name in error]
        if relevant:
            cards.append(
                ValidationCard(
                    title=file_name,
                    state="error",
                    detail=" | ".join(relevant),
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


def _render_validation_banner(error_count: int) -> None:
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


def _render_run_status_strip(bundle: RunBundle) -> None:
    provider_count = len(bundle.ranked_providers)
    objection_count = len(bundle.objections)
    script_count = len(bundle.scripts)
    retrieval_count = len(bundle.retrieval_debug)
    st.markdown(
        f"""
        <div class="status-strip">
          <div class="status-node">
            <span>Market ranked</span>
            <strong>{provider_count} providers</strong>
          </div>
          <div class="status-node">
            <span>Objections drafted</span>
            <strong>{objection_count} handlers</strong>
          </div>
          <div class="status-node">
            <span>Scripts built</span>
            <strong>{script_count} scripts</strong>
          </div>
          <div class="status-node">
            <span>Retrieval traces</span>
            <strong>{retrieval_count} providers</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_objection_card(item: ObjectionView) -> None:
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


def _render_provider_detail(
    provider: RankedProviderView,
    objections: list[ObjectionView],
    scripts: list[MeetingScriptView],
) -> None:
    st.markdown(
        f"""
        <article class="artifact-card">
          <div class="artifact-head">
            <span class="artifact-pill">{provider.provider_id}</span>
            <span class="artifact-pill">{provider.physician_name}</span>
            <span class="artifact-pill">{provider.institution}</span>
            <span class="artifact-pill">score {provider.score:.2f}</span>
          </div>
          <p>{provider.rationale}</p>
          <div class="artifact-foot">
            objections: {len(objections)} | scripts: {len(scripts)}
          </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def _render_downloads(bundle: RunBundle) -> None:
    downloads = _artifact_downloads(bundle.run_dir)
    if not downloads:
        st.caption("No artifact files are available to download for this run.")
        return

    st.markdown('<div class="control-block">Artifact Downloads</div>', unsafe_allow_html=True)
    columns = st.columns(min(3, len(downloads)))
    for index, artifact in enumerate(downloads):
        with columns[index % len(columns)]:
            st.download_button(
                label=artifact.label,
                data=artifact.path.read_bytes(),
                file_name=artifact.path.name,
                mime="text/plain",
                use_container_width=True,
            )


def _render_validation_cards(summary: ValidationSummary) -> None:
    cards = _validation_cards(summary)
    st.markdown('<div class="control-block">Validation Breakdown</div>', unsafe_allow_html=True)
    markup = "".join(
        f"""
        <article class="validation-card validation-card--{card.state}">
          <span>{card.title}</span>
          <strong>{card.state.upper()}</strong>
          <p>{card.detail}</p>
        </article>
        """
        for card in cards
    )
    st.markdown(f'<div class="validation-grid">{markup}</div>', unsafe_allow_html=True)


def _matching_objections(
    bundle: RunBundle,
    provider_id: str,
) -> list[ObjectionView]:
    return [item for item in bundle.objections if item.provider_id == provider_id]


def _matching_scripts(
    bundle: RunBundle,
    provider_id: str,
) -> list[MeetingScriptView]:
    return [item for item in bundle.scripts if item.provider_id == provider_id]


def _render_provider_focus(bundle: RunBundle, provider: RankedProviderView) -> None:
    objections = _matching_objections(bundle, provider.provider_id)
    scripts = _matching_scripts(bundle, provider.provider_id)
    _render_provider_detail(provider, objections, scripts)

    if objections:
        st.markdown("#### Objection Stack")
        for item in objections:
            _render_objection_card(item)
    else:
        st.caption("No objections were generated for this provider.")

    if scripts:
        st.markdown("#### Meeting Script")
        for item in scripts:
            _render_script_card(item)
    else:
        st.caption("No meeting scripts were generated for this provider.")


def _render_retrieval_trace(bundle: RunBundle) -> None:
    if not bundle.retrieval_debug:
        st.caption("Retrieval diagnostics were not written for this run.")
        return

    for item in bundle.retrieval_debug:
        label = f"{item.provider_id}: {item.query_text or 'query unavailable'}"
        with st.expander(label):
            rows = _retrieval_rows(item)
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.caption("No retrieval hits were recorded for this provider.")


def _filtered_providers(
    bundle: RunBundle,
    query: str,
    min_score: float,
) -> list[RankedProviderView]:
    needle = query.strip().lower()
    providers = [provider for provider in bundle.ranked_providers if provider.score >= min_score]
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


def _build_run_overrides(settings: Settings) -> tuple[SettingsOverride, RunControls, bool]:
    st.markdown('<div class="control-block">Run Configuration</div>', unsafe_allow_html=True)
    with st.form("run-pipeline-form", clear_on_submit=False):
        primary_left, primary_right = st.columns(2)
        with primary_left:
            market_csv = st.text_input("Market CSV", value=str(settings.market_csv))
            crm_csv = st.text_input("CRM CSV", value=str(settings.crm_csv))
            kb_markdown = st.text_input(
                "Knowledge Base Markdown",
                value=str(settings.kb_markdown),
            )
        with primary_right:
            output_dir = st.text_input("Output Directory", value=str(settings.output_dir))
            strict_citations = st.checkbox(
                "Strict Citations",
                value=settings.output.strict_citations,
            )
            enforce_low_confidence = st.checkbox("Fail on Low Confidence", value=False)
            confidence_floor = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                disabled=not enforce_low_confidence,
            )

        with st.expander("Advanced Runtime Settings"):
            advanced_left, advanced_right = st.columns(2)
            with advanced_left:
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
                top_k = st.number_input(
                    "Top K",
                    min_value=1,
                    value=settings.rag.top_k,
                    step=1,
                )
            with advanced_right:
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
                backoff_seconds = st.number_input(
                    "Retry Backoff Seconds",
                    min_value=0.0,
                    value=float(settings.rag.backoff_seconds),
                    step=0.25,
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


def _render_bundle(bundle: RunBundle) -> None:
    render_metric_cards(bundle)
    _render_run_status_strip(bundle)
    _render_metadata(bundle)

    overview_tab, provider_tab, retrieval_tab, validation_tab, downloads_tab = st.tabs(
        ["Overview", "Provider Focus", "Retrieval", "Validation", "Downloads"]
    )

    with overview_tab:
        st.subheader("Ranked Providers")
        if bundle.ranked_providers:
            st.dataframe(
                _provider_rows(bundle.ranked_providers),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No ranked providers were written for this run.")

        st.subheader("Objection Handlers")
        if bundle.objections:
            for item in bundle.objections[:5]:
                _render_objection_card(item)
        else:
            st.caption("No objection handlers were generated for this run.")

    with provider_tab:
        provider_query = st.text_input("Search providers", value="", key=f"search-{bundle.run_dir}")
        min_score = st.slider(
            "Minimum score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key=f"score-{bundle.run_dir}",
        )
        filtered = _filtered_providers(bundle, provider_query, min_score)
        if filtered:
            provider_ids = [provider.provider_id for provider in filtered]
            selected_provider = st.selectbox(
                "Provider",
                options=provider_ids,
                key=f"provider-{bundle.run_dir}",
            )
            provider = next(
                provider for provider in filtered if provider.provider_id == selected_provider
            )
            _render_provider_focus(bundle, provider)
        else:
            st.warning("No providers match the current filters.")

    with retrieval_tab:
        _render_retrieval_trace(bundle)

    with validation_tab:
        summary = validate_run_summary(bundle.run_dir)
        _render_validation_cards(summary)
        if summary.errors:
            st.markdown("#### Raw issues")
            for error in summary.errors:
                st.markdown(f"- {error}")

    with downloads_tab:
        _render_downloads(bundle)
        st.subheader("Run Directory")
        st.code(str(bundle.run_dir), language="text")


def render_run_page(config_path: Path) -> None:
    _render_section_intro(
        "Run Pipeline",
        "Launch a new pipeline run with a clean primary form. Advanced retrieval and "
        "runtime controls stay tucked away until you need them.",
    )
    settings = load_default_settings(config_path)
    summaries = summarize_runs(settings.output_dir)
    last_bundle = _load_latest_bundle(summaries)

    st.markdown(
        f"""
        <div class="insight-banner">
          <strong>Config source</strong>
          <span>{config_path}</span>
          <strong>Latest run</strong>
          <span>{last_bundle.run_dir.name if last_bundle else "none yet"}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overrides, controls, submitted = _build_run_overrides(settings)
    bundle = last_bundle

    if submitted:
        try:
            with st.status("Running pipeline", expanded=True) as status:
                st.write("Loading inputs and applying in-memory overrides.")
                st.write("Ranking providers and building retrieval context.")
                run_dir = run_pipeline_from_ui(
                    config_path,
                    settings_overrides=overrides,
                    controls=controls,
                )
                st.write("Reading generated artifacts and validation state.")
                bundle = load_run_bundle(run_dir)
                status.update(label="Pipeline completed", state="complete", expanded=False)
            if bundle is not None:
                _remember_last_run(bundle.run_dir)
                st.success(f"Run completed: {bundle.run_dir}")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.error(f"Run failed: {exc}")
            st.markdown(
                """
                <div class="empty-state">
                  <strong>Troubleshooting</strong>
                  <p>
                    Check input paths, Ollama credentials, and whether the local embedding
                    endpoint is reachable. The app does not rewrite config files, so only
                    the current form submission is affected.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

    if bundle is None:
        st.markdown(
            """
            <div class="empty-state">
              <strong>No rendered run yet</strong>
              <p>
                Run the pipeline to populate the workspace. If the configured inputs are
                missing, the backend will generate mock data before the run starts.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    _render_validation_banner(len(bundle.validation_errors))
    _render_bundle(bundle)


def render_runs_page(config_path: Path) -> None:
    _render_section_intro(
        "Browse Runs",
        "Review historical runs as a field desk: pick a run, narrow the provider list, "
        "then jump into objections, scripts, retrieval, and downloads.",
    )
    settings = load_default_settings(config_path)
    summaries = summarize_runs(settings.output_dir)
    if not summaries:
        st.markdown(
            """
            <div class="empty-state">
              <strong>No runs in the archive</strong>
              <p>
                The browse workspace activates once `outputs/run_*` folders exist. Launch a
                run from the pipeline page or validate an existing run directory manually.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    options = {_format_run_label(summary): summary for summary in summaries}
    selected_label = st.selectbox("Select Run", list(options.keys()))
    selected = options[selected_label]
    bundle = load_run_bundle(selected.run_dir)
    _remember_last_run(bundle.run_dir)

    _render_validation_banner(len(bundle.validation_errors))
    _render_bundle(bundle)


def render_validate_page(config_path: Path) -> None:
    _render_section_intro(
        "Validate Outputs",
        "Inspect the output contract file-by-file before you circulate a run or use it as a "
        "reference artifact.",
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
        st.markdown(
            """
            <div class="empty-state">
              <strong>Select a run first</strong>
              <p>
                Choose a run from the archive or enter a manual path to validate a folder on disk.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    summary = validate_run_summary(Path(candidate))
    _render_validation_banner(len(summary.errors))
    _render_validation_cards(summary)

    if summary.is_valid:
        st.success(f"Validation passed for `{candidate}`.")
    else:
        st.error(f"Validation failed for `{candidate}`.")
        for error in summary.errors:
            st.markdown(f"- {error}")

    downloads = _artifact_downloads(Path(candidate))
    if downloads:
        st.markdown("#### Available artifacts")
        columns = st.columns(min(3, len(downloads)))
        for index, artifact in enumerate(downloads):
            with columns[index % len(columns)]:
                st.download_button(
                    label=artifact.label,
                    data=artifact.path.read_bytes(),
                    file_name=artifact.path.name,
                    mime="text/plain",
                    use_container_width=True,
                )


def render_about_page(config_path: Path) -> None:
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
          <div class="artifact-foot">Persistence: last successful run is remembered in session</div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    load_dotenv()
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":material/analytics:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    stylesheet = load_stylesheet()
    if stylesheet:
        st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)

    settings = load_default_settings(DEFAULT_CONFIG_PATH)
    summaries = summarize_runs(settings.output_dir)

    _render_hero()
    st.caption(APP_SUBTITLE)

    config_path_text = st.sidebar.text_input("Config Path", value=str(DEFAULT_CONFIG_PATH))
    config_path = Path(config_path_text)
    _render_sidebar_status(config_path, summaries)
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
