import numpy as np

from tempus_copilot.rag.embed import HashEmbeddingClient


def test_hash_embedding_client_is_deterministic() -> None:
    client = HashEmbeddingClient(dimension=8)
    vectors_a = client.embed_texts(["alpha", "beta"])
    vectors_b = client.embed_texts(["alpha", "beta"])
    assert vectors_a.shape == (2, 8)
    assert np.array_equal(vectors_a, vectors_b)
