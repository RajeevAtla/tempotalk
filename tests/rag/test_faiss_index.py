import numpy as np

from tempus_copilot.rag.faiss_index import FaissIndex


def test_faiss_index_roundtrip_query() -> None:
    index = FaissIndex(dimension=3)
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    metadata = [{"id": "a"}, {"id": "b"}]
    index.add(vectors=vectors, metadata=metadata)
    hits = index.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)
    assert len(hits) == 1
    assert hits[0]["id"] == "a"


def test_faiss_index_query_with_scores_returns_distance() -> None:
    index = FaissIndex(dimension=2)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index.add(vectors=vectors, metadata=[{"id": "x"}, {"id": "y"}])
    hits = index.query_with_scores(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert hits[0]["metadata"]["id"] == "x"
    assert isinstance(hits[0]["distance"], float)
