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
    chunk_id: str
    source: str
    text: str


class RetrievedItem(TypedDict):
    chunk_id: str
    source: str
    distance: float


class RetrievalRow(TypedDict):
    provider_id: str
    query_text: str
    retrieved: list[RetrievedItem]


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


class ObjectionPayload(TypedDict):
    schema_version: str
    objections: list[ObjectionRow]


class MeetingPayload(TypedDict):
    schema_version: str
    scripts: list[ScriptRow]


class RetrievalPayload(TypedDict):
    schema_version: str
    retrieval_debug: list[RetrievalRow]


@dataclass(frozen=True)
class BamlHashes:
    schema_hash: str
    prompt_hash: str


class EmbedTexts(Protocol):
    def __call__(self, texts: list[str]) -> np.ndarray: ...


def ensure_inputs(settings: Settings) -> None:
    if settings.market_csv.exists() and settings.crm_csv.exists() and settings.kb_markdown.exists():
        return
    generate_mock_data(
        output_dir=settings.market_csv.parent,
        seed=settings.mock_seed,
        scale=settings.mock_scale,
    )


def write_toml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        toml_dump(payload, fh)


def baml_source_path() -> Path:
    return Path("baml_src/sales_copilot.baml")


def compute_baml_hashes(path: Path | None = None) -> BamlHashes:
    source_path = baml_source_path() if path is None else path
    if not source_path.exists():
        empty_hash = sha256(b"").hexdigest()
        return BamlHashes(schema_hash=empty_hash, prompt_hash=empty_hash)
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
    if not strict_citations:
        return citations, confidence
    allowed_set = set(allowed)
    sanitized = [item for item in citations if item in allowed_set]
    if len(sanitized) == len(citations):
        return sanitized, confidence
    return sanitized, max(0.0, confidence - 0.25)


def build_kb_chunks(kb_docs: list[KBDocument], settings: Settings) -> list[KBChunk]:
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
    chunk_vectors = embed_texts([item["text"] for item in kb_chunks])
    if chunk_vectors.size == 0:
        chunk_vectors = np.zeros(
            (len(kb_chunks), embedding_dimension),
            dtype=np.float32,
        )
    if chunk_vectors.ndim != 2:
        raise ValueError("Embedding matrix must be 2-dimensional")
    return chunk_vectors


def chunk_metadata_rows(kb_chunks: list[KBChunk]) -> list[dict[str, str]]:
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
    checksum_payload = "|".join([ranked_text, objection_text, script_text]).encode("utf-8")
    return sha256(checksum_payload).hexdigest()


class RetrievedSearchHit(TypedDict):
    metadata: dict[str, str]
    distance: float
