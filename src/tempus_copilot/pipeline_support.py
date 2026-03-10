"""Shared helper types and utilities for pipeline orchestration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
from tomli_w import dump as toml_dump

from scripts.generate_mock_data import generate_mock_data
from tempus_copilot.config import Settings
from tempus_copilot.models import KBDocument
from tempus_copilot.rag.chunking import chunk_text


class KBChunk(TypedDict):
    """Represents one chunk of knowledge-base text."""

    chunk_id: str
    source: str
    text: str


class RetrievedItem(TypedDict):
    """Represents one retrieval hit written to debug output."""

    chunk_id: str
    source: str
    distance: float


class RetrievalRow(TypedDict):
    """Represents retrieval debug data for one provider query."""

    provider_id: str
    query_text: str
    retrieved: list[RetrievedItem]


class ObjectionRow(TypedDict):
    """Represents one objection handler row in the output payload."""

    provider_id: str
    concern: str
    response: str
    supporting_metrics: list[str]
    citations: list[str]
    confidence: float


class ScriptRow(TypedDict):
    """Represents one meeting script row in the output payload."""

    provider_id: str
    tumor_focus: str
    script: str
    citations: list[str]
    confidence: float


class ObjectionPayload(TypedDict):
    """Represents the TOML payload for objection handler output."""

    schema_version: str
    objections: list[ObjectionRow]


class MeetingPayload(TypedDict):
    """Represents the TOML payload for meeting script output."""

    schema_version: str
    scripts: list[ScriptRow]


class RetrievalPayload(TypedDict):
    """Represents the TOML payload for retrieval debug output."""

    schema_version: str
    retrieval_debug: list[RetrievalRow]


@dataclass(frozen=True)
class BamlHashes:
    """Stores separate hashes for BAML schema and prompt content."""

    schema_hash: str
    prompt_hash: str


class EmbedTexts(Protocol):
    """Defines the embedding callback contract used by the pipeline."""

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Embed ordered texts into a vector matrix.

        Args:
            texts: Ordered input texts to embed.

        Returns:
            A two-dimensional embedding matrix.
        """
        ...


def ensure_inputs(settings: Settings) -> None:
    """Ensures required input fixtures exist for the current run.

    Args:
        settings: Application settings containing input paths and mock settings.
    """
    if settings.market_csv.exists() and settings.crm_csv.exists() and settings.kb_markdown.exists():
        return
    generate_mock_data(
        output_dir=settings.market_csv.parent,
        seed=settings.mock_seed,
        scale=settings.mock_scale,
    )


def write_toml(path: Path, payload: Mapping[str, object]) -> None:
    """Writes a mapping to TOML, creating parent directories if needed.

    Args:
        path: Destination TOML path.
        payload: Top-level TOML payload to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        toml_dump(payload, fh)


def baml_source_path() -> Path:
    """Returns the canonical checked-in BAML source path."""

    return Path("baml_src/sales_copilot.baml")


def compute_baml_hashes(path: Path | None = None) -> BamlHashes:
    """Computes separate hashes for BAML schema lines and prompt blocks.

    Args:
        path: Optional explicit BAML source path.

    Returns:
        Hashes describing the effective schema and prompt text.
    """
    source_path = baml_source_path() if path is None else path
    if not source_path.exists():
        empty_hash = sha256(b"").hexdigest()
        return BamlHashes(schema_hash=empty_hash, prompt_hash=empty_hash)
    # Hash schema and prompt text separately so prompt-only edits remain visible in metadata.
    text = source_path.read_text(encoding="utf-8")
    schema_lines: list[str] = []
    prompt_blocks = re.findall(r'prompt\s+#"(.*?)"#', text, flags=re.DOTALL)
    in_prompt = False
    for line in text.splitlines():
        if 'prompt #"' in line:
            in_prompt = True
            continue
        if in_prompt and '"#' in line:
            in_prompt = False
            continue
        if not in_prompt:
            schema_lines.append(line)
    return BamlHashes(
        schema_hash=sha256("\n".join(schema_lines).encode("utf-8")).hexdigest(),
        prompt_hash=sha256("\n".join(prompt_blocks).encode("utf-8")).hexdigest(),
    )


def extract_metrics(text: str) -> list[str]:
    """Extracts a small set of metric-like values from free text.

    Args:
        text: Source text to scan.

    Returns:
        Deduplicated metric strings in first-seen order.
    """
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


def build_query_text(provider_id: str, tumor_focus: str, concern: str) -> str:
    """Builds the retrieval query used for one provider.

    Args:
        provider_id: Provider identifier.
        tumor_focus: Provider tumor focus.
        concern: Primary concern to address.

    Returns:
        Retrieval query text.
    """
    return (
        f"Provider {provider_id} with tumor focus {tumor_focus}. "
        f"Address concern: {concern}."
    )


def enforce_citations(
    citations: list[str],
    allowed: list[str],
    confidence: float,
    strict_citations: bool,
) -> tuple[list[str], float]:
    """Sanitizes citations and adjusts confidence in strict mode.

    Args:
        citations: Citations produced by the generation step.
        allowed: Retrieval-backed citation identifiers.
        confidence: Model-reported confidence score.
        strict_citations: Whether unsupported citations should be filtered.

    Returns:
        Sanitized citations and the possibly adjusted confidence score.
    """
    if not strict_citations:
        return citations, confidence
    allowed_set = set(allowed)
    # Strict mode keeps only retrieval-backed citations and lightly penalizes unsupported ones.
    sanitized = [item for item in citations if item in allowed_set]
    if len(sanitized) == len(citations):
        return sanitized, confidence
    return sanitized, max(0.0, confidence - 0.25)


def build_kb_chunks(kb_docs: list[KBDocument], settings: Settings) -> list[KBChunk]:
    """Splits KB documents into retrieval chunks.

    Args:
        kb_docs: Loaded knowledge-base documents.
        settings: Application settings with chunking controls.

    Returns:
        Chunk records ready for embedding and indexing.
    """
    kb_chunks: list[KBChunk] = []
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
    return kb_chunks


def build_chunk_vectors(
    kb_chunks: list[KBChunk],
    *,
    embedding_dimension: int,
    embed_texts: EmbedTexts,
) -> np.ndarray:
    """Embeds KB chunks into a two-dimensional vector matrix.

    Args:
        kb_chunks: Chunk records to embed.
        embedding_dimension: Fallback width for empty embedding results.
        embed_texts: Embedding callback.

    Returns:
        Two-dimensional embedding matrix.

    Raises:
        ValueError: If the embedding callback returns a non-2D matrix.
    """
    chunk_vectors = embed_texts([item["text"] for item in kb_chunks])
    if chunk_vectors.size == 0:
        # Preserve the configured embedding width even when the KB is empty.
        chunk_vectors = np.zeros(
            (len(kb_chunks), embedding_dimension),
            dtype=np.float32,
        )
    if chunk_vectors.ndim != 2:
        raise ValueError("Embedding matrix must be 2-dimensional")
    return chunk_vectors


def chunk_metadata_rows(kb_chunks: list[KBChunk]) -> list[dict[str, str]]:
    """Builds FAISS metadata rows from KB chunks.

    Args:
        kb_chunks: Chunk records to project into metadata rows.

    Returns:
        Plain metadata dictionaries aligned with the chunk vectors.
    """
    return [
        {
            "chunk_id": item["chunk_id"],
            "source": item["source"],
            "text": item["text"],
        }
        for item in kb_chunks
    ]


def build_retrieval_row(
    provider_id: str,
    query_text: str,
    retrieved: list[RetrievedSearchHit],
) -> RetrievalRow:
    """Builds one retrieval debug row from scored hits.

    Args:
        provider_id: Provider identifier.
        query_text: Retrieval query text used for the provider.
        retrieved: Scored retrieval hits.

    Returns:
        Retrieval debug row ready for TOML serialization.
    """
    return {
        "provider_id": provider_id,
        "query_text": query_text,
        "retrieved": [
            {
                "chunk_id": item["metadata"].get("chunk_id", ""),
                "source": item["metadata"].get("source", ""),
                "distance": float(item["distance"]),
            }
            for item in retrieved
        ],
    }


def checksum_output_texts(ranked_text: str, objection_text: str, script_text: str) -> str:
    """Computes the canonical checksum for generated text artifacts.

    Args:
        ranked_text: Ranked providers TOML contents.
        objection_text: Objection handlers TOML contents.
        script_text: Meeting scripts TOML contents.

    Returns:
        SHA-256 checksum over the combined payload text.
    """
    # This must stay in lockstep with output_schema.compute_output_checksum.
    checksum_payload = "|".join([ranked_text, objection_text, script_text]).encode("utf-8")
    return sha256(checksum_payload).hexdigest()


class RetrievedSearchHit(TypedDict):
    """Represents a scored retrieval result consumed by pipeline helpers."""

    metadata: dict[str, str]
    distance: float
