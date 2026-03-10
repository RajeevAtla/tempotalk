"""Unit tests for the FAISS-backed retrieval index."""

import numpy as np
import pytest

from tempus_copilot.rag import faiss_index as faiss_index_module
from tempus_copilot.rag.faiss_index import FaissIndex


def test_faiss_index_roundtrip_query() -> None:
    """Return the nearest metadata row for a matching query vector."""
    index = FaissIndex(dimension=3)
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    metadata = [{"id": "a"}, {"id": "b"}]
    index.add(vectors=vectors, metadata=metadata)
    hits = index.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)
    assert len(hits) == 1
    assert hits[0]["id"] == "a"


def test_faiss_index_query_with_scores_returns_distance() -> None:
    """Expose distances alongside metadata rows."""
    index = FaissIndex(dimension=2)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index.add(vectors=vectors, metadata=[{"id": "x"}, {"id": "y"}])
    hits = index.query_with_scores(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert hits[0]["metadata"]["id"] == "x"
    assert isinstance(hits[0]["distance"], float)


def test_faiss_index_validation_branches() -> None:
    """Reject invalid dimensions and mismatched vector metadata shapes."""
    with pytest.raises(ValueError):
        FaissIndex(dimension=0)

    index = FaissIndex(dimension=2)
    with pytest.raises(ValueError):
        index.add(vectors=np.array([1.0, 0.0], dtype=np.float32), metadata=[{"id": "x"}])
    with pytest.raises(ValueError):
        index.add(
            vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            metadata=[{"id": "x"}],
        )


def test_faiss_index_uses_dtype_conversion_and_2d_query_path() -> None:
    """Coerce float64 inputs and support already-batched query vectors."""
    index = FaissIndex(dimension=2)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    index.add(vectors=vectors, metadata=[{"id": "x"}, {"id": "y"}])
    hits = index.query_with_scores(np.array([[1.0, 0.0]], dtype=np.float64), top_k=1)
    assert hits[0]["metadata"]["id"] == "x"


def test_faiss_index_missing_ctor_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise when the FAISS module lacks the expected constructor."""

    class NoCtor:
        """Stand in for a broken FAISS module."""

        pass

    monkeypatch.setattr(faiss_index_module, "_load_faiss", lambda: NoCtor())
    with pytest.raises(RuntimeError):
        FaissIndex(dimension=2)
