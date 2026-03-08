from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tomli_w import dump as toml_dump

from scripts.generate_mock_data import generate_mock_data
from tempus_copilot.config import Settings
from tempus_copilot.ingest.crm import load_crm_notes
from tempus_copilot.ingest.kb import load_kb_markdown
from tempus_copilot.ingest.market import load_market_intelligence
from tempus_copilot.llm.baml_adapter import BamlAdapter
from tempus_copilot.models import PipelineResult
from tempus_copilot.ranking.score import rank_providers


def _ensure_inputs(settings: Settings) -> None:
    if settings.market_csv.exists() and settings.crm_csv.exists() and settings.kb_markdown.exists():
        return
    generate_mock_data(
        output_dir=settings.market_csv.parent,
        seed=settings.mock_seed,
        scale=settings.mock_scale,
    )


def _write_toml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        toml_dump(payload, fh)


def run_pipeline(settings: Settings) -> PipelineResult:
    _ensure_inputs(settings)
    providers = load_market_intelligence(settings.market_csv)
    notes = load_crm_notes(settings.crm_csv)
    kb_docs = load_kb_markdown(settings.kb_markdown)
    ranked = rank_providers(
        providers=providers,
        crm_notes=notes,
        weights=settings.ranking_weights,
    )
    run_dir = settings.output_dir / datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = run_dir / "ranked_providers.toml"
    objection_path = run_dir / "objection_handlers.toml"
    script_path = run_dir / "meeting_scripts.toml"
    metadata_path = run_dir / "run_metadata.toml"
    _write_toml(
        ranked_path,
        {
            "providers": [
                {
                    "provider_id": item.provider_id,
                    "physician_name": item.physician_name,
                    "institution": item.institution,
                    "score": item.score,
                    "rationale": item.rationale,
                }
                for item in ranked
            ]
        },
    )
    adapter = BamlAdapter()
    objection_payload = {
        "objections": [
            {
                "provider_id": provider.provider_id,
                "response": adapter.generate_objection_handler(
                    provider=provider,
                    notes=[note for note in notes if note.provider_id == provider.provider_id],
                    kb=kb_docs,
                ),
            }
            for provider in providers
        ]
    }
    meeting_payload = {
        "scripts": [
            {
                "provider_id": provider.provider_id,
                "script": adapter.generate_meeting_script(provider=provider, kb=kb_docs),
            }
            for provider in providers
        ]
    }
    _write_toml(objection_path, objection_payload)
    _write_toml(script_path, meeting_payload)
    _write_toml(
        metadata_path,
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "provider_count": len(providers),
            "note_count": len(notes),
            "kb_doc_count": len(kb_docs),
            "generation_model": settings.models.generation_model,
            "embedding_model": settings.models.embedding_model,
        },
    )
    return PipelineResult(
        run_dir=run_dir,
        ranked_providers_path=ranked_path,
        objection_handlers_path=objection_path,
        meeting_scripts_path=script_path,
        metadata_path=metadata_path,
    )
