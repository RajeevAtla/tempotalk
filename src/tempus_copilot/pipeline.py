from __future__ import annotations

import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Mapping, TypedDict

import numpy as np
from tomli_w import dump as toml_dump

from scripts.generate_mock_data import generate_mock_data
from tempus_copilot.config import Settings
from tempus_copilot.ingest.crm import load_crm_notes
from tempus_copilot.ingest.kb import load_kb_markdown
from tempus_copilot.ingest.market import load_market_intelligence
from tempus_copilot.llm.baml_adapter import GenerationClient, get_default_generation_client
from tempus_copilot.models import PipelineResult
from tempus_copilot.rag.chunking import chunk_text
from tempus_copilot.rag.embed import (
    EmbeddingClient,
    FallbackEmbeddingClient,
    GeminiEmbeddingClient,
    HashEmbeddingClient,
)
from tempus_copilot.rag.faiss_index import FaissIndex
from tempus_copilot.ranking.score import rank_providers

SCHEMA_VERSION = "1.0.0"


class ObjectionRow(TypedDict):
    provider_id: str
    concern: str
    response: str
    supporting_metrics: list[str]
    citations: list[str]
    confidence: float


class ScriptRow(TypedDict):
    provider_id: str
    tumor_focus: str
    script: str
    citations: list[str]
    confidence: float


class RetrievedItem(TypedDict):
    chunk_id: str
    source: str
    distance: float


class RetrievalRow(TypedDict):
    provider_id: str
    query_text: str
    retrieved: list[RetrievedItem]


class ObjectionPayload(TypedDict):
    schema_version: str
    objections: list[ObjectionRow]


class MeetingPayload(TypedDict):
    schema_version: str
    scripts: list[ScriptRow]


class RetrievalPayload(TypedDict):
    schema_version: str
    retrieval_debug: list[RetrievalRow]


def _ensure_inputs(settings: Settings) -> None:
    if settings.market_csv.exists() and settings.crm_csv.exists() and settings.kb_markdown.exists():
        return
    generate_mock_data(
        output_dir=settings.market_csv.parent,
        seed=settings.mock_seed,
        scale=settings.mock_scale,
    )


def _write_toml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        toml_dump(payload, fh)


def _baml_source_path() -> Path:
    return Path("baml_src/sales_copilot.baml")


def _compute_baml_hashes() -> tuple[str, str]:
    path = _baml_source_path()
    if not path.exists():
        empty_hash = sha256(b"").hexdigest()
        return empty_hash, empty_hash
    text = path.read_text(encoding="utf-8")
    schema_lines: list[str] = []
    prompt_blocks = re.findall(r'prompt\s+#"(.*?)"#', text, flags=re.DOTALL)
    in_prompt = False
    for line in text.splitlines():
        if "prompt #\"" in line:
            in_prompt = True
            continue
        if in_prompt and "\"#" in line:
            in_prompt = False
            continue
        if not in_prompt:
            schema_lines.append(line)
    schema_hash = sha256("\n".join(schema_lines).encode("utf-8")).hexdigest()
    prompt_hash = sha256("\n".join(prompt_blocks).encode("utf-8")).hexdigest()
    return schema_hash, prompt_hash


def _extract_metrics(text: str) -> list[str]:
    patterns = [
        r"\b\d+(?:\.\d+)?%\b",
        r"\b\d+(?:\.\d+)?\b",
        r"\b\d+\s*(?:day|days|hour|hours)\b",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    deduped: list[str] = []
    for value in matches:
        if value not in deduped:
            deduped.append(value)
    return deduped[:5]


def _build_query_text(provider_id: str, tumor_focus: str, concern: str) -> str:
    return (
        f"Provider {provider_id} with tumor focus {tumor_focus}. "
        f"Address concern: {concern}."
    )


def _default_embedding_client(settings: Settings) -> EmbeddingClient:
    fallback = HashEmbeddingClient(dimension=settings.rag.embedding_dimension)
    if settings.models.embedding_provider == "google":
        try:
            primary = GeminiEmbeddingClient(
                model=settings.models.embedding_model,
                request_retries=settings.rag.request_retries,
                backoff_seconds=settings.rag.backoff_seconds,
            )
            return FallbackEmbeddingClient(primary=primary, fallback=fallback)
        except RuntimeError:
            return fallback
    return fallback


def _enforce_citations(
    citations: list[str],
    allowed: list[str],
    confidence: float,
    strict_citations: bool,
) -> tuple[list[str], float]:
    if not strict_citations:
        return citations, confidence
    allowed_set = set(allowed)
    sanitized = [item for item in citations if item in allowed_set]
    if len(sanitized) == len(citations):
        return sanitized, confidence
    return sanitized, max(0.0, confidence - 0.25)


def run_pipeline(
    settings: Settings,
    embedding_client: EmbeddingClient | None = None,
    generation_client: GenerationClient | None = None,
    strict_citations: bool | None = None,
    fail_on_low_confidence: float | None = None,
) -> PipelineResult:
    _ensure_inputs(settings)
    providers = load_market_intelligence(settings.market_csv)
    notes = load_crm_notes(settings.crm_csv)
    kb_docs = load_kb_markdown(settings.kb_markdown)
    ranked = rank_providers(
        providers=providers,
        crm_notes=notes,
        weights=settings.ranking_weights,
        calibration=settings.ranking_calibration,
    )
    embedder = embedding_client or _default_embedding_client(settings)
    generator = generation_client or get_default_generation_client()
    enforce_strict = (
        settings.output.strict_citations if strict_citations is None else strict_citations
    )

    kb_chunks: list[dict[str, str]] = []
    for doc in kb_docs:
        chunks = chunk_text(
            text=doc.text,
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
        )
        for idx, text in enumerate(chunks):
            kb_chunks.append(
                {
                    "chunk_id": f"{doc.source}:{idx}",
                    "source": doc.source,
                    "text": text,
                }
            )
    chunk_vectors = embedder.embed_texts([item["text"] for item in kb_chunks])
    if chunk_vectors.size == 0:
        chunk_vectors = np.zeros(
            (len(kb_chunks), settings.rag.embedding_dimension),
            dtype=np.float32,
        )
    if chunk_vectors.ndim != 2:
        raise ValueError("Embedding matrix must be 2-dimensional")
    index = FaissIndex(dimension=int(chunk_vectors.shape[1]))
    index.add(vectors=chunk_vectors, metadata=kb_chunks)

    run_dir = settings.output_dir / datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = run_dir / "ranked_providers.toml"
    objection_path = run_dir / "objection_handlers.toml"
    script_path = run_dir / "meeting_scripts.toml"
    retrieval_debug_path = run_dir / "retrieval_debug.toml"
    metadata_path = run_dir / "run_metadata.toml"
    _write_toml(
        ranked_path,
        {
            "schema_version": SCHEMA_VERSION,
            "providers": [
                {
                    "provider_id": item.provider_id,
                    "physician_name": item.physician_name,
                    "institution": item.institution,
                    "score": item.score,
                    "rationale": item.rationale,
                    "factor_scores": item.factor_scores,
                    "calibration_terms": item.calibration_terms,
                    "factor_contributions": item.factor_contributions,
                }
                for item in ranked
            ]
        },
    )
    objection_rows: list[ObjectionRow] = []
    script_rows: list[ScriptRow] = []
    retrieval_rows: list[RetrievalRow] = []
    for provider in providers:
        provider_notes = [note for note in notes if note.provider_id == provider.provider_id]
        concern = provider_notes[0].concern_type if provider_notes else "general"
        query_text = _build_query_text(provider.provider_id, provider.tumor_focus, concern)
        query_vector = embedder.embed_texts([query_text])[0]
        retrieved = index.query_with_scores(query_vector, top_k=settings.rag.top_k)
        retrieved_meta = [item["metadata"] for item in retrieved]
        context = "\n\n".join(item["text"] for item in retrieved_meta)
        citations = [item["chunk_id"] for item in retrieved_meta]
        metrics = _extract_metrics(context)
        objection = generator.generate_objection_handler(
            provider_id=provider.provider_id,
            concern=concern,
            kb_context=context,
            citation_ids=citations,
            observed_metrics=metrics,
        )
        script = generator.generate_meeting_script(
            provider_id=provider.provider_id,
            tumor_focus=provider.tumor_focus,
            kb_context=context,
            citation_ids=citations,
        )
        objection_citations, objection_confidence = _enforce_citations(
            citations=objection.citations,
            allowed=citations,
            confidence=objection.confidence,
            strict_citations=enforce_strict,
        )
        script_citations, script_confidence = _enforce_citations(
            citations=script.citations,
            allowed=citations,
            confidence=script.confidence,
            strict_citations=enforce_strict,
        )
        objection_rows.append(
            {
                "provider_id": objection.provider_id,
                "concern": objection.concern,
                "response": objection.response,
                "supporting_metrics": objection.supporting_metrics,
                "citations": objection_citations,
                "confidence": objection_confidence,
            }
        )
        script_rows.append(
            {
                "provider_id": script.provider_id,
                "tumor_focus": script.tumor_focus,
                "script": script.script,
                "citations": script_citations,
                "confidence": script_confidence,
            }
        )
        retrieval_rows.append(
            {
                "provider_id": provider.provider_id,
                "query_text": query_text,
                "retrieved": [
                    {
                        "chunk_id": str(item["metadata"].get("chunk_id", "")),
                        "source": str(item["metadata"].get("source", "")),
                        "distance": float(item["distance"]),
                    }
                    for item in retrieved
                ],
            }
        )
    objection_payload: ObjectionPayload = {
        "schema_version": SCHEMA_VERSION,
        "objections": objection_rows,
    }
    meeting_payload: MeetingPayload = {
        "schema_version": SCHEMA_VERSION,
        "scripts": script_rows,
    }
    retrieval_payload: RetrievalPayload = {
        "schema_version": SCHEMA_VERSION,
        "retrieval_debug": retrieval_rows,
    }
    _write_toml(objection_path, objection_payload)
    _write_toml(script_path, meeting_payload)
    _write_toml(retrieval_debug_path, retrieval_payload)
    checksum_payload = "|".join(
        [
            ranked_path.read_text(encoding="utf-8"),
            objection_path.read_text(encoding="utf-8"),
            script_path.read_text(encoding="utf-8"),
        ]
    ).encode("utf-8")
    checksum = sha256(checksum_payload).hexdigest()
    baml_schema_hash, baml_prompt_hash = _compute_baml_hashes()
    _write_toml(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "provider_count": len(providers),
            "note_count": len(notes),
            "kb_doc_count": len(kb_docs),
            "kb_chunk_count": len(kb_chunks),
            "generation_model": settings.models.generation_model,
            "embedding_model": settings.models.embedding_model,
            "output_checksum_sha256": checksum,
            "baml_schema_sha256": baml_schema_hash,
            "baml_prompt_sha256": baml_prompt_hash,
        },
    )
    if fail_on_low_confidence is not None:
        all_confidences = [float(row["confidence"]) for row in objection_rows] + [
            float(row["confidence"]) for row in script_rows
        ]
        if any(value < fail_on_low_confidence for value in all_confidences):
            raise ValueError(
                f"Confidence threshold violated: threshold={fail_on_low_confidence}"
            )
    return PipelineResult(
        run_dir=run_dir,
        ranked_providers_path=ranked_path,
        objection_handlers_path=objection_path,
        meeting_scripts_path=script_path,
        retrieval_debug_path=retrieval_debug_path,
        metadata_path=metadata_path,
    )
