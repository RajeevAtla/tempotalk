"""Pipeline support helper tests."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from tempus_copilot.config import load_settings
from tempus_copilot.pipeline_support import (
    KBChunk,
    baml_source_path,
    build_chunk_vectors,
    build_query_text,
    build_retrieval_row,
    checksum_output_texts,
    chunk_metadata_rows,
    compute_baml_hashes,
    ensure_inputs,
    write_toml,
)


def test_ensure_inputs_is_noop_when_files_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies input generation is skipped when all configured files already exist."""
    data_dir = tmp_path / "inputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ("market_intelligence.csv", "crm_notes.csv", "product_kb.md"):
        (data_dir / file_name).write_text("x", encoding="utf-8")
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={
            "market_csv": data_dir / "market_intelligence.csv",
            "crm_csv": data_dir / "crm_notes.csv",
            "kb_markdown": data_dir / "product_kb.md",
        }
    )

    def fail_generate(output_dir: Path, seed: int, scale: int) -> None:
        """Fail generate.
        
        Args:
            output_dir: Filesystem path for output dir.
            seed: Seed.
            scale: Scale.
        """
        raise AssertionError(f"unexpected generation {output_dir} {seed} {scale}")

    monkeypatch.setattr("tempus_copilot.pipeline_support.generate_mock_data", fail_generate)
    ensure_inputs(settings)


def test_ensure_inputs_generates_files_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies missing inputs trigger support-layer mock-data generation."""
    data_dir = tmp_path / "missing"
    settings = load_settings(Path("config/defaults.toml")).model_copy(
        update={
            "market_csv": data_dir / "market_intelligence.csv",
            "crm_csv": data_dir / "crm_notes.csv",
            "kb_markdown": data_dir / "product_kb.md",
        }
    )
    called = {"value": False}

    def fake_generate(output_dir: Path, seed: int, scale: int) -> None:
        """Fake generate.
        
        Args:
            output_dir: Filesystem path for output dir.
            seed: Seed.
            scale: Scale.
        """
        called["value"] = True
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "market_intelligence.csv").write_text("market", encoding="utf-8")
        (output_dir / "crm_notes.csv").write_text("crm", encoding="utf-8")
        (output_dir / "product_kb.md").write_text("kb", encoding="utf-8")

    monkeypatch.setattr("tempus_copilot.pipeline_support.generate_mock_data", fake_generate)
    ensure_inputs(settings)
    assert called["value"] is True


def test_write_toml_and_checksum_helpers_round_trip(tmp_path: Path) -> None:
    """Verifies TOML writing and checksum helpers produce stable outputs."""
    output_path = tmp_path / "sample.toml"
    write_toml(output_path, {"schema_version": "1.0.0", "values": ["a", "b"]})
    assert output_path.exists()
    checksum = checksum_output_texts("ranked", "objection", "script")
    expected = sha256(b"ranked|objection|script").hexdigest()
    assert checksum == expected


def test_baml_source_path_default_and_explicit_hash_computation(tmp_path: Path) -> None:
    """Verifies default and explicit BAML hash computations both produce values."""
    source_path = baml_source_path()
    assert source_path == Path("baml_src/sales_copilot.baml")
    default_hashes = compute_baml_hashes()
    assert default_hashes.schema_hash
    assert default_hashes.prompt_hash

    custom_baml = tmp_path / "custom.baml"
    custom_baml.write_text(
        'class Demo {\n  value string\n}\n\nfunction Test() -> Demo {\n  prompt #"\nhello\n"#\n}\n',
        encoding="utf-8",
    )
    hashes = compute_baml_hashes(custom_baml)
    assert hashes.schema_hash
    assert hashes.prompt_hash
    assert hashes.schema_hash != hashes.prompt_hash


def test_build_query_chunk_and_retrieval_helpers() -> None:
    """Verifies retrieval helper builders return the expected row shapes."""
    assert (
        build_query_text("P001", "Lung", "general")
        == "Provider P001 with tumor focus Lung. Address concern: general."
    )
    chunks: list[KBChunk] = [
        {"chunk_id": "product_kb.md:0", "source": "product_kb.md", "text": "turnaround 8 days"}
    ]
    metadata_rows = chunk_metadata_rows(chunks)
    assert metadata_rows == [
        {
            "chunk_id": "product_kb.md:0",
            "source": "product_kb.md",
            "text": "turnaround 8 days",
        }
    ]
    retrieval_row = build_retrieval_row(
        provider_id="P001",
        query_text="query",
        retrieved=[
            {
                "metadata": {"chunk_id": "product_kb.md:0", "source": "product_kb.md"},
                "distance": 0.25,
            }
        ],
    )
    assert retrieval_row["retrieved"][0]["distance"] == 0.25


def test_build_chunk_vectors_returns_default_zeros_for_empty_embeddings() -> None:
    """Verifies empty embedding results are replaced with zero vectors."""
    chunks: list[KBChunk] = [
        {"chunk_id": "product_kb.md:0", "source": "product_kb.md", "text": "chunk text"}
    ]

    def empty_embed_texts(texts: list[str]) -> np.ndarray:
        """Empty embed texts.
        
        Args:
            texts: Texts.
        
        Returns:
            Computed result.
        """
        assert texts == ["chunk text"]
        return np.empty((0, 0), dtype=np.float32)

    vectors = build_chunk_vectors(
        chunks,
        embedding_dimension=4,
        embed_texts=empty_embed_texts,
    )
    assert vectors.shape == (1, 4)
