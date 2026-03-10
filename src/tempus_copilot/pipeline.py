"""End-to-end orchestration for the TempoTalk ranking and generation pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from scripts.generate_mock_data import generate_mock_data
from tempus_copilot.config import Settings
from tempus_copilot.ingest.crm import load_crm_notes
from tempus_copilot.ingest.kb import load_kb_markdown
from tempus_copilot.ingest.market import load_market_intelligence
from tempus_copilot.llm.baml_adapter import GenerationClient, get_default_generation_client
from tempus_copilot.models import CRMNote, PipelineResult, ProviderRecord, RankedProvider
from tempus_copilot.pipeline_support import (
    KBChunk,
    MeetingPayload,
    ObjectionPayload,
    ObjectionRow,
    RetrievalPayload,
    RetrievalRow,
    ScriptRow,
    build_chunk_vectors,
    build_kb_chunks,
    build_query_text,
    build_retrieval_row,
    checksum_output_texts,
    chunk_metadata_rows,
    compute_baml_hashes,
    enforce_citations,
    extract_metrics,
    write_toml,
)
from tempus_copilot.rag.embed import EmbeddingClient, OllamaEmbeddingClient
from tempus_copilot.rag.faiss_index import FaissIndex
from tempus_copilot.ranking.score import rank_providers

SCHEMA_VERSION = "1.0.0"


def _baml_source_path() -> Path:
    """Returns the checked-in BAML source path used for metadata hashes."""

    return Path("baml_src/sales_copilot.baml")


def _compute_baml_hashes() -> tuple[str, str]:
    """Returns schema and prompt hashes for the canonical BAML source."""

    hashes = compute_baml_hashes(_baml_source_path())
    return hashes.schema_hash, hashes.prompt_hash


def _extract_metrics(text: str) -> list[str]:
    """Extracts a short list of metric strings from retrieved context."""

    return extract_metrics(text)


def _build_query_text(provider_id: str, tumor_focus: str, concern: str) -> str:
    """Builds the retrieval query for one provider."""

    return build_query_text(provider_id, tumor_focus, concern)


def _ensure_inputs(settings: Settings) -> None:
    """Ensures required local input files exist before the run starts.

    Args:
        settings: Application settings containing input paths and mock settings.
    """
    # The prototype can self-seed its local fixtures, but only when any required input is missing.
    if settings.market_csv.exists() and settings.crm_csv.exists() and settings.kb_markdown.exists():
        return
    generate_mock_data(
        output_dir=settings.market_csv.parent,
        seed=settings.mock_seed,
        scale=settings.mock_scale,
    )


def _enforce_citations(
    citations: list[str],
    allowed: list[str],
    confidence: float,
    strict_citations: bool,
) -> tuple[list[str], float]:
    """Applies citation policy to generated citations and confidence."""

    return enforce_citations(
        citations=citations,
        allowed=allowed,
        confidence=confidence,
        strict_citations=strict_citations,
    )


def _default_embedding_client(settings: Settings) -> EmbeddingClient:
    """Builds the default embedding client from settings.

    Args:
        settings: Application settings containing embedding runtime config.

    Returns:
        Configured embedding client.

    Raises:
        ValueError: If the configured embedding provider is unsupported.
    """
    provider = settings.models.embedding_provider.lower().strip()
    if provider != "ollama":
        raise ValueError("Only 'ollama' embedding_provider is supported")
    return OllamaEmbeddingClient(
        model=settings.models.embedding_model,
        request_retries=settings.rag.request_retries,
        backoff_seconds=settings.rag.backoff_seconds,
    )


def _resolve_runtime_clients(
    settings: Settings,
    embedding_client: EmbeddingClient | None,
    generation_client: GenerationClient | None,
) -> tuple[EmbeddingClient, GenerationClient]:
    """Resolves injected or default runtime clients for the pipeline.

    Args:
        settings: Application settings containing runtime config.
        embedding_client: Optional injected embedding client.
        generation_client: Optional injected generation client.

    Returns:
        Embedding and generation clients in runtime order.
    """
    # Injected clients keep tests local while defaults enforce the repo's runtime policy.
    embedder = embedding_client or _default_embedding_client(settings)
    generator = generation_client or get_default_generation_client(
        generation_provider=settings.models.generation_provider,
        generation_model=settings.models.generation_model,
        request_retries=settings.rag.request_retries,
        backoff_seconds=settings.rag.backoff_seconds,
    )
    return embedder, generator


def _build_index(
    settings: Settings,
    kb_chunks: Sequence[KBChunk],
    embedder: EmbeddingClient,
) -> FaissIndex:
    """Builds the retrieval index for the current KB chunks.

    Args:
        settings: Application settings with retrieval controls.
        kb_chunks: Chunked knowledge-base records.
        embedder: Embedding client used to vectorize chunks.

    Returns:
        Populated FAISS index.
    """
    chunk_vectors = build_chunk_vectors(
        list(kb_chunks),
        embedding_dimension=settings.rag.embedding_dimension,
        embed_texts=embedder.embed_texts,
    )
    index = FaissIndex(dimension=int(chunk_vectors.shape[1]))
    index.add(vectors=chunk_vectors, metadata=chunk_metadata_rows(list(kb_chunks)))
    return index


def _write_ranked_providers(run_dir: Path, ranked: Sequence[RankedProvider]) -> Path:
    """Writes ranked provider output for the current run.

    Args:
        run_dir: Directory for the current pipeline run.
        ranked: Ranked provider records to serialize.

    Returns:
        Path to the written ranked provider artifact.
    """
    ranked_path = run_dir / "ranked_providers.toml"
    write_toml(
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
            ],
        },
    )
    return ranked_path


def _run_generation(
    *,
    providers: Sequence[ProviderRecord],
    notes: Sequence[CRMNote],
    settings: Settings,
    embedder: EmbeddingClient,
    generator: GenerationClient,
    index: FaissIndex,
    strict_citations: bool,
) -> tuple[list[ObjectionRow], list[ScriptRow], list[RetrievalRow]]:
    """Runs retrieval-backed generation for every provider.

    Args:
        providers: Providers to generate artifacts for.
        notes: CRM notes available for concern detection.
        settings: Application settings with retrieval controls.
        embedder: Embedding client for provider queries.
        generator: Generation client for objection and script artifacts.
        index: Retrieval index over KB chunks.
        strict_citations: Whether strict citation filtering is enabled.

    Returns:
        Objection rows, script rows, and retrieval debug rows.
    """
    objection_rows: list[ObjectionRow] = []
    script_rows: list[ScriptRow] = []
    retrieval_rows: list[RetrievalRow] = []
    for provider in providers:
        # Missing CRM history is still a valid path, so generation falls back to a generic concern.
        provider_notes = [note for note in notes if note.provider_id == provider.provider_id]
        concern = provider_notes[0].concern_type if provider_notes else "general"
        query_text = build_query_text(provider.provider_id, provider.tumor_focus, concern)
        query_vector = embedder.embed_texts([query_text])[0]
        retrieved = index.query_with_scores(query_vector, top_k=settings.rag.top_k)
        retrieved_meta = [item["metadata"] for item in retrieved]
        context = "\n\n".join(item["text"] for item in retrieved_meta)
        citations = [item["chunk_id"] for item in retrieved_meta]
        metrics = extract_metrics(context)
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
        objection_citations, objection_confidence = enforce_citations(
            citations=objection.citations,
            allowed=citations,
            confidence=objection.confidence,
            strict_citations=strict_citations,
        )
        script_citations, script_confidence = enforce_citations(
            citations=script.citations,
            allowed=citations,
            confidence=script.confidence,
            strict_citations=strict_citations,
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
            build_retrieval_row(
                provider_id=provider.provider_id,
                query_text=query_text,
                retrieved=retrieved,
            )
        )
    return objection_rows, script_rows, retrieval_rows


def _write_generation_outputs(
    run_dir: Path,
    objection_rows: list[ObjectionRow],
    script_rows: list[ScriptRow],
    retrieval_rows: list[RetrievalRow],
) -> tuple[Path, Path, Path]:
    """Writes objection, script, and retrieval debug artifacts.

    Args:
        run_dir: Directory for the current pipeline run.
        objection_rows: Serialized objection handler rows.
        script_rows: Serialized meeting script rows.
        retrieval_rows: Serialized retrieval debug rows.

    Returns:
        Paths to the written objection, script, and retrieval artifacts.
    """
    objection_path = run_dir / "objection_handlers.toml"
    script_path = run_dir / "meeting_scripts.toml"
    retrieval_debug_path = run_dir / "retrieval_debug.toml"
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
    write_toml(objection_path, objection_payload)
    write_toml(script_path, meeting_payload)
    write_toml(retrieval_debug_path, retrieval_payload)
    return objection_path, script_path, retrieval_debug_path


def _write_metadata(
    *,
    run_dir: Path,
    ranked_path: Path,
    objection_path: Path,
    script_path: Path,
    settings: Settings,
    provider_count: int,
    note_count: int,
    kb_doc_count: int,
    kb_chunk_count: int,
) -> Path:
    """Writes run metadata and checksum information.

    Args:
        run_dir: Directory for the current pipeline run.
        ranked_path: Ranked providers artifact path.
        objection_path: Objection handlers artifact path.
        script_path: Meeting scripts artifact path.
        settings: Application settings for model metadata.
        provider_count: Number of provider input records.
        note_count: Number of CRM note records.
        kb_doc_count: Number of KB documents loaded.
        kb_chunk_count: Number of KB chunks indexed.

    Returns:
        Path to the written metadata artifact.
    """
    metadata_path = run_dir / "run_metadata.toml"
    # Metadata records reproducibility details, but the output checksum stays anchored to the
    # primary deliverables that downstream validators and reviewers consume.
    checksum = checksum_output_texts(
        ranked_path.read_text(encoding="utf-8"),
        objection_path.read_text(encoding="utf-8"),
        script_path.read_text(encoding="utf-8"),
    )
    baml_hashes = compute_baml_hashes(_baml_source_path())
    write_toml(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "provider_count": provider_count,
            "note_count": note_count,
            "kb_doc_count": kb_doc_count,
            "kb_chunk_count": kb_chunk_count,
            "generation_model": settings.models.generation_model,
            "embedding_model": settings.models.embedding_model,
            "output_checksum_sha256": checksum,
            "baml_schema_sha256": baml_hashes.schema_hash,
            "baml_prompt_sha256": baml_hashes.prompt_hash,
        },
    )
    return metadata_path


def run_pipeline(
    settings: Settings,
    embedding_client: EmbeddingClient | None = None,
    generation_client: GenerationClient | None = None,
    strict_citations: bool | None = None,
    fail_on_low_confidence: float | None = None,
) -> PipelineResult:
    """Runs the full ranking, retrieval, generation, and write pipeline.

    Args:
        settings: Application settings for the current run.
        embedding_client: Optional injected embedding client.
        generation_client: Optional injected generation client.
        strict_citations: Optional override for citation enforcement.
        fail_on_low_confidence: Optional minimum confidence threshold.

    Returns:
        Paths to the artifacts written by the run.

    Raises:
        ValueError: If the configured confidence threshold is violated.
    """
    # The pipeline writes ranked output first, then generation artifacts, so failed runs still
    # leave enough state behind for debugging.
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
    embedder, generator = _resolve_runtime_clients(settings, embedding_client, generation_client)
    enforce_strict = (
        settings.output.strict_citations if strict_citations is None else strict_citations
    )
    kb_chunks = build_kb_chunks(kb_docs, settings)
    index = _build_index(settings, kb_chunks, embedder)

    run_dir = settings.output_dir / datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = _write_ranked_providers(run_dir, ranked)
    objection_rows, script_rows, retrieval_rows = _run_generation(
        providers=providers,
        notes=notes,
        settings=settings,
        embedder=embedder,
        generator=generator,
        index=index,
        strict_citations=enforce_strict,
    )
    objection_path, script_path, retrieval_debug_path = _write_generation_outputs(
        run_dir=run_dir,
        objection_rows=objection_rows,
        script_rows=script_rows,
        retrieval_rows=retrieval_rows,
    )
    metadata_path = _write_metadata(
        run_dir=run_dir,
        ranked_path=ranked_path,
        objection_path=objection_path,
        script_path=script_path,
        settings=settings,
        provider_count=len(providers),
        note_count=len(notes),
        kb_doc_count=len(kb_docs),
        kb_chunk_count=len(kb_chunks),
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
