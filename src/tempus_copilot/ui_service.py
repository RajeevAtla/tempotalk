from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, cast

from tempus_copilot.config import Settings, load_settings
from tempus_copilot.output_schema import REQUIRED_TOP_LEVEL, parse_toml, validate_run_outputs
from tempus_copilot.pipeline import run_pipeline

DEFAULT_CONFIG_PATH = Path("config/defaults.toml")
ARTIFACT_LABELS = {
    "ranked_providers.toml": "Ranked Providers",
    "objection_handlers.toml": "Objection Handlers",
    "meeting_scripts.toml": "Meeting Scripts",
    "retrieval_debug.toml": "Retrieval Debug",
    "run_metadata.toml": "Run Metadata",
}


class ProviderTableRow(TypedDict):
    provider_id: str
    physician_name: str
    institution: str
    score: float
    rationale: str


class ObjectionTableRow(TypedDict):
    provider_id: str
    concern: str
    response: str
    supporting_metrics: list[str]
    citations: list[str]
    confidence: float


class MeetingScriptTableRow(TypedDict):
    provider_id: str
    tumor_focus: str
    script: str
    citations: list[str]
    confidence: float


class RetrievalHitRow(TypedDict):
    chunk_id: str
    source: str
    distance: float


class RetrievalTableRow(TypedDict):
    provider_id: str
    query_text: str
    retrieved: list[RetrievalHitRow]


class RunMetadataView(TypedDict, total=False):
    schema_version: str
    generated_at_utc: str
    provider_count: int
    note_count: int
    kb_doc_count: int
    kb_chunk_count: int
    generation_model: str
    embedding_model: str
    output_checksum_sha256: str
    baml_schema_sha256: str
    baml_prompt_sha256: str


@dataclass(frozen=True)
class RankedProviderView:
    provider_id: str
    physician_name: str
    institution: str
    score: float
    rationale: str
    factor_scores: dict[str, float]
    calibration_terms: dict[str, float]
    factor_contributions: dict[str, float]


@dataclass(frozen=True)
class ObjectionView:
    provider_id: str
    concern: str
    response: str
    supporting_metrics: list[str]
    citations: list[str]
    confidence: float


@dataclass(frozen=True)
class MeetingScriptView:
    provider_id: str
    tumor_focus: str
    script: str
    citations: list[str]
    confidence: float


@dataclass(frozen=True)
class RetrievalHitView:
    chunk_id: str
    source: str
    distance: float


@dataclass(frozen=True)
class RetrievalDebugView:
    provider_id: str
    query_text: str
    retrieved: list[RetrievalHitView]


@dataclass(frozen=True)
class RunBundle:
    run_dir: Path
    ranked_providers: list[RankedProviderView]
    objections: list[ObjectionView]
    scripts: list[MeetingScriptView]
    retrieval_debug: list[RetrievalDebugView]
    metadata: RunMetadataView
    validation_errors: list[str] = field(default_factory=list)
    validation_report: ValidationReport | None = None
    artifacts: list[ArtifactFileView] = field(default_factory=list)


@dataclass(frozen=True)
class ArtifactFileView:
    file_name: str
    label: str
    path: Path
    exists: bool

    @property
    def download_name(self) -> str:
        return f"{self.path.parent.name}_{self.file_name}"


@dataclass(frozen=True)
class ValidationFileStatus:
    file_name: str
    label: str
    exists: bool
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return self.exists and not self.errors


@dataclass(frozen=True)
class ValidationReport:
    run_dir: Path
    file_statuses: list[ValidationFileStatus]
    checksum_error: str | None
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class SettingsOverride:
    market_csv: Path | None = None
    crm_csv: Path | None = None
    kb_markdown: Path | None = None
    output_dir: Path | None = None
    generation_model: str | None = None
    embedding_model: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    top_k: int | None = None
    request_retries: int | None = None
    backoff_seconds: float | None = None
    strict_citations: bool | None = None
    patient_volume_weight: float | None = None
    clinical_fit_weight: float | None = None
    objection_urgency_weight: float | None = None
    recency_weight: float | None = None


@dataclass(frozen=True)
class RunControls:
    strict_citations: bool | None = None
    fail_on_low_confidence: float | None = None


@dataclass(frozen=True)
class ValidationSummary:
    run_dir: Path
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    generated_at_utc: str
    provider_count: int
    generation_model: str
    embedding_model: str
    validation_errors: list[str]
    objection_count: int = 0
    script_count: int = 0


@dataclass(frozen=True)
class LoadedRunSummary:
    run_dir: Path
    metadata: RunMetadataView
    providers: list[ProviderTableRow]
    objections: list[ObjectionTableRow]
    meeting_scripts: list[MeetingScriptTableRow]
    retrieval_debug: list[RetrievalTableRow]
    validation: ValidationSummary


def load_ui_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    return load_settings(config_path)


def load_default_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    return load_ui_settings(config_path)


def apply_settings_overrides(settings: Settings, overrides: SettingsOverride) -> Settings:
    updated = settings

    path_updates: dict[str, Path] = {}
    if overrides.market_csv is not None:
        path_updates["market_csv"] = overrides.market_csv
    if overrides.crm_csv is not None:
        path_updates["crm_csv"] = overrides.crm_csv
    if overrides.kb_markdown is not None:
        path_updates["kb_markdown"] = overrides.kb_markdown
    if overrides.output_dir is not None:
        path_updates["output_dir"] = overrides.output_dir
    if path_updates:
        updated = updated.model_copy(update=path_updates)

    model_updates: dict[str, str] = {}
    if overrides.generation_model is not None:
        model_updates["generation_model"] = overrides.generation_model
    if overrides.embedding_model is not None:
        model_updates["embedding_model"] = overrides.embedding_model
    if model_updates:
        updated = updated.model_copy(
            update={"models": updated.models.model_copy(update=model_updates)}
        )

    rag_updates: dict[str, int | float] = {}
    if overrides.chunk_size is not None:
        rag_updates["chunk_size"] = overrides.chunk_size
    if overrides.chunk_overlap is not None:
        rag_updates["chunk_overlap"] = overrides.chunk_overlap
    if overrides.top_k is not None:
        rag_updates["top_k"] = overrides.top_k
    if overrides.request_retries is not None:
        rag_updates["request_retries"] = overrides.request_retries
    if overrides.backoff_seconds is not None:
        rag_updates["backoff_seconds"] = overrides.backoff_seconds
    if rag_updates:
        updated = updated.model_copy(update={"rag": updated.rag.model_copy(update=rag_updates)})

    ranking_updates: dict[str, float] = {}
    if overrides.patient_volume_weight is not None:
        ranking_updates["patient_volume"] = overrides.patient_volume_weight
    if overrides.clinical_fit_weight is not None:
        ranking_updates["clinical_fit"] = overrides.clinical_fit_weight
    if overrides.objection_urgency_weight is not None:
        ranking_updates["objection_urgency"] = overrides.objection_urgency_weight
    if overrides.recency_weight is not None:
        ranking_updates["recency"] = overrides.recency_weight
    if ranking_updates:
        updated = updated.model_copy(
            update={
                "ranking_weights": updated.ranking_weights.model_copy(update=ranking_updates)
            }
        )

    if overrides.strict_citations is not None:
        updated = updated.model_copy(
            update={
                "output": updated.output.model_copy(
                    update={"strict_citations": overrides.strict_citations}
                )
            }
        )

    return updated


def run_pipeline_from_ui(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    settings_overrides: SettingsOverride | None = None,
    controls: RunControls | None = None,
) -> Path:
    settings = load_ui_settings(config_path)
    if settings_overrides is not None:
        settings = apply_settings_overrides(settings, settings_overrides)
    run_controls = controls or RunControls()
    result = run_pipeline(
        settings,
        strict_citations=run_controls.strict_citations,
        fail_on_low_confidence=run_controls.fail_on_low_confidence,
    )
    return result.run_dir


def discover_run_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    run_dirs = [
        path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("run_")
    ]
    run_dirs.sort(key=lambda path: path.name, reverse=True)
    return run_dirs


def most_recent_run_dir(output_dir: Path) -> Path | None:
    run_dirs = discover_run_dirs(output_dir)
    if not run_dirs:
        return None
    return run_dirs[0]


def load_run_summary(run_dir: Path) -> LoadedRunSummary:
    return LoadedRunSummary(
        run_dir=run_dir,
        metadata=_load_metadata(run_dir),
        providers=_load_providers(run_dir),
        objections=_load_objections(run_dir),
        meeting_scripts=_load_meeting_scripts(run_dir),
        retrieval_debug=_load_retrieval_debug(run_dir),
        validation=validate_run_summary(run_dir),
    )


def validate_run_summary(run_dir: Path) -> ValidationSummary:
    return ValidationSummary(run_dir=run_dir, errors=validate_run_outputs(run_dir))


def load_validation_report(run_dir: Path) -> ValidationReport:
    errors = validate_run_outputs(run_dir)
    checksum_error: str | None = None
    file_statuses: list[ValidationFileStatus] = []
    for file_name in REQUIRED_TOP_LEVEL:
        path = run_dir / file_name
        file_errors = [
            error
            for error in errors
            if error.startswith(f"Missing output file: {file_name}")
            or error.startswith(f"{file_name} missing key:")
        ]
        file_statuses.append(
            ValidationFileStatus(
                file_name=file_name,
                label=ARTIFACT_LABELS.get(file_name, file_name),
                exists=path.exists(),
                errors=file_errors,
            )
        )
    for error in errors:
        if error.startswith("Checksum mismatch"):
            checksum_error = error
            break
    return ValidationReport(
        run_dir=run_dir,
        file_statuses=file_statuses,
        checksum_error=checksum_error,
        errors=errors,
    )


def list_artifact_files(run_dir: Path) -> list[ArtifactFileView]:
    return [
        ArtifactFileView(
            file_name=file_name,
            label=ARTIFACT_LABELS.get(file_name, file_name),
            path=run_dir / file_name,
            exists=(run_dir / file_name).exists(),
        )
        for file_name in REQUIRED_TOP_LEVEL
    ]


def load_run_bundle(run_dir: Path) -> RunBundle:
    summary = load_run_summary(run_dir)
    validation_report = load_validation_report(run_dir)
    return RunBundle(
        run_dir=summary.run_dir,
        ranked_providers=[
            RankedProviderView(
                provider_id=row["provider_id"],
                physician_name=row["physician_name"],
                institution=row["institution"],
                score=row["score"],
                rationale=row["rationale"],
                factor_scores={},
                calibration_terms={},
                factor_contributions={},
            )
            for row in summary.providers
        ],
        objections=[
            ObjectionView(
                provider_id=row["provider_id"],
                concern=row["concern"],
                response=row["response"],
                supporting_metrics=row["supporting_metrics"],
                citations=row["citations"],
                confidence=row["confidence"],
            )
            for row in summary.objections
        ],
        scripts=[
            MeetingScriptView(
                provider_id=row["provider_id"],
                tumor_focus=row["tumor_focus"],
                script=row["script"],
                citations=row["citations"],
                confidence=row["confidence"],
            )
            for row in summary.meeting_scripts
        ],
        retrieval_debug=[
            RetrievalDebugView(
                provider_id=row["provider_id"],
                query_text=row["query_text"],
                retrieved=[
                    RetrievalHitView(
                        chunk_id=hit["chunk_id"],
                        source=hit["source"],
                        distance=hit["distance"],
                    )
                    for hit in row["retrieved"]
                ],
            )
            for row in summary.retrieval_debug
        ],
        metadata=summary.metadata,
        validation_errors=summary.validation.errors,
        validation_report=validation_report,
        artifacts=list_artifact_files(run_dir),
    )


def summarize_runs(output_dir: Path) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for run_dir in discover_run_dirs(output_dir):
        loaded = load_run_summary(run_dir)
        validation_errors = loaded.validation.errors
        summaries.append(
            RunSummary(
                run_dir=loaded.run_dir,
                generated_at_utc=loaded.metadata.get("generated_at_utc", ""),
                provider_count=loaded.metadata.get("provider_count", len(loaded.providers)),
                generation_model=loaded.metadata.get("generation_model", ""),
                embedding_model=loaded.metadata.get("embedding_model", ""),
                validation_errors=validation_errors,
                objection_count=len(loaded.objections),
                script_count=len(loaded.meeting_scripts),
            )
        )
    return summaries


def _load_metadata(run_dir: Path) -> RunMetadataView:
    payload = _load_payload(run_dir / "run_metadata.toml")
    if not payload:
        return {}
    metadata: RunMetadataView = {}

    schema_version = payload.get("schema_version")
    if isinstance(schema_version, str):
        metadata["schema_version"] = schema_version
    generated_at_utc = payload.get("generated_at_utc")
    if isinstance(generated_at_utc, str):
        metadata["generated_at_utc"] = generated_at_utc
    provider_count = _coerce_metadata_int(payload.get("provider_count"))
    if provider_count is not None:
        metadata["provider_count"] = provider_count
    note_count = _coerce_metadata_int(payload.get("note_count"))
    if note_count is not None:
        metadata["note_count"] = note_count
    kb_doc_count = _coerce_metadata_int(payload.get("kb_doc_count"))
    if kb_doc_count is not None:
        metadata["kb_doc_count"] = kb_doc_count
    kb_chunk_count = _coerce_metadata_int(payload.get("kb_chunk_count"))
    if kb_chunk_count is not None:
        metadata["kb_chunk_count"] = kb_chunk_count
    generation_model = payload.get("generation_model")
    if isinstance(generation_model, str):
        metadata["generation_model"] = generation_model
    embedding_model = payload.get("embedding_model")
    if isinstance(embedding_model, str):
        metadata["embedding_model"] = embedding_model
    output_checksum = payload.get("output_checksum_sha256")
    if isinstance(output_checksum, str):
        metadata["output_checksum_sha256"] = output_checksum
    baml_schema_hash = payload.get("baml_schema_sha256")
    if isinstance(baml_schema_hash, str):
        metadata["baml_schema_sha256"] = baml_schema_hash
    baml_prompt_hash = payload.get("baml_prompt_sha256")
    if isinstance(baml_prompt_hash, str):
        metadata["baml_prompt_sha256"] = baml_prompt_hash

    return metadata


def _load_providers(run_dir: Path) -> list[ProviderTableRow]:
    payload = _load_payload(run_dir / "ranked_providers.toml")
    providers = _get_mapping_list(payload, "providers")
    return [
        {
            "provider_id": _get_str(item, "provider_id"),
            "physician_name": _get_str(item, "physician_name"),
            "institution": _get_str(item, "institution"),
            "score": _get_float(item, "score"),
            "rationale": _get_str(item, "rationale"),
        }
        for item in providers
    ]


def _load_objections(run_dir: Path) -> list[ObjectionTableRow]:
    payload = _load_payload(run_dir / "objection_handlers.toml")
    objections = _get_mapping_list(payload, "objections")
    return [
        {
            "provider_id": _get_str(item, "provider_id"),
            "concern": _get_str(item, "concern"),
            "response": _get_str(item, "response"),
            "supporting_metrics": _get_string_list(item, "supporting_metrics"),
            "citations": _get_string_list(item, "citations"),
            "confidence": _get_float(item, "confidence"),
        }
        for item in objections
    ]


def _load_meeting_scripts(run_dir: Path) -> list[MeetingScriptTableRow]:
    payload = _load_payload(run_dir / "meeting_scripts.toml")
    scripts = _get_mapping_list(payload, "scripts")
    return [
        {
            "provider_id": _get_str(item, "provider_id"),
            "tumor_focus": _get_str(item, "tumor_focus"),
            "script": _get_str(item, "script"),
            "citations": _get_string_list(item, "citations"),
            "confidence": _get_float(item, "confidence"),
        }
        for item in scripts
    ]


def _load_retrieval_debug(run_dir: Path) -> list[RetrievalTableRow]:
    payload = _load_payload(run_dir / "retrieval_debug.toml")
    rows = _get_mapping_list(payload, "retrieval_debug")
    return [
        {
            "provider_id": _get_str(item, "provider_id"),
            "query_text": _get_str(item, "query_text"),
            "retrieved": [
                {
                    "chunk_id": _get_str(hit, "chunk_id"),
                    "source": _get_str(hit, "source"),
                    "distance": _get_float(hit, "distance"),
                }
                for hit in _get_mapping_list(item, "retrieved")
            ],
        }
        for item in rows
    ]


def _load_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return parse_toml(path)


def _get_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}


def _get_mapping_list(mapping: dict[str, object], key: str) -> list[dict[str, object]]:
    value = mapping.get(key)
    if not isinstance(value, list):
        return []
    rows: list[dict[str, object]] = []
    for item in value:
        row = _get_mapping(item)
        if row:
            rows.append(row)
    return rows


def _get_str(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key, "")
    return value if isinstance(value, str) else str(value)


def _get_float(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key, 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _coerce_metadata_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _get_string_list(mapping: dict[str, object], key: str) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
