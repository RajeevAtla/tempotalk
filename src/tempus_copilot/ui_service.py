from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from tempus_copilot.config import Settings, load_settings
from tempus_copilot.output_schema import parse_toml, validate_run_outputs
from tempus_copilot.pipeline import run_pipeline

DEFAULT_CONFIG_PATH = Path("config/defaults.toml")


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
    metadata: dict[str, object]
    providers: list[ProviderTableRow]
    objections: list[ObjectionTableRow]
    meeting_scripts: list[MeetingScriptTableRow]
    retrieval_debug: list[RetrievalTableRow]
    validation: ValidationSummary


def load_ui_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    return load_settings(config_path)


def apply_settings_overrides(settings: Settings, overrides: SettingsOverride) -> Settings:
    path_updates: dict[str, object] = {}
    if overrides.market_csv is not None:
        path_updates["market_csv"] = overrides.market_csv
    if overrides.crm_csv is not None:
        path_updates["crm_csv"] = overrides.crm_csv
    if overrides.kb_markdown is not None:
        path_updates["kb_markdown"] = overrides.kb_markdown
    if overrides.output_dir is not None:
        path_updates["output_dir"] = overrides.output_dir

    model_updates: dict[str, object] = {}
    if overrides.generation_model is not None:
        model_updates["generation_model"] = overrides.generation_model
    if overrides.embedding_model is not None:
        model_updates["embedding_model"] = overrides.embedding_model

    rag_updates: dict[str, object] = {}
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

    ranking_updates: dict[str, object] = {}
    if overrides.patient_volume_weight is not None:
        ranking_updates["patient_volume"] = overrides.patient_volume_weight
    if overrides.clinical_fit_weight is not None:
        ranking_updates["clinical_fit"] = overrides.clinical_fit_weight
    if overrides.objection_urgency_weight is not None:
        ranking_updates["objection_urgency"] = overrides.objection_urgency_weight
    if overrides.recency_weight is not None:
        ranking_updates["recency"] = overrides.recency_weight

    output_updates: dict[str, object] = {}
    if overrides.strict_citations is not None:
        output_updates["strict_citations"] = overrides.strict_citations

    updated = settings
    if path_updates:
        updated = updated.model_copy(update=path_updates)
    if model_updates:
        updated = updated.model_copy(
            update={"models": updated.models.model_copy(update=model_updates)}
        )
    if rag_updates:
        updated = updated.model_copy(update={"rag": updated.rag.model_copy(update=rag_updates)})
    if ranking_updates:
        updated = updated.model_copy(
            update={
                "ranking_weights": updated.ranking_weights.model_copy(update=ranking_updates)
            }
        )
    if output_updates:
        updated = updated.model_copy(
            update={"output": updated.output.model_copy(update=output_updates)}
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
    run_options = controls or RunControls()
    result = run_pipeline(
        settings,
        strict_citations=run_options.strict_citations,
        fail_on_low_confidence=run_options.fail_on_low_confidence,
    )
    return result.run_dir


def discover_run_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    runs = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.startswith("run_")
    ]
    runs.sort(key=lambda path: path.name, reverse=True)
    return runs


def load_run_summary(run_dir: Path) -> RunSummary:
    return RunSummary(
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


def _load_metadata(run_dir: Path) -> dict[str, object]:
    metadata_path = run_dir / "run_metadata.toml"
    if not metadata_path.exists():
        return {}
    return parse_toml(metadata_path)


def _load_providers(run_dir: Path) -> list[ProviderTableRow]:
    payload = _load_payload(run_dir / "ranked_providers.toml")
    providers = _get_list(payload, "providers")
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
    objections = _get_list(payload, "objections")
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
    scripts = _get_list(payload, "scripts")
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
    rows = _get_list(payload, "retrieval_debug")
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
                for hit in _get_list(item, "retrieved")
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


def _get_list(mapping: dict[str, object], key: str) -> list[dict[str, object]]:
    value = mapping.get(key)
    if not isinstance(value, list):
        return []
    return [_get_mapping(item) for item in value]


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


def _get_string_list(mapping: dict[str, object], key: str) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
