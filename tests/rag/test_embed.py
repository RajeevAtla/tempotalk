import httpx
import numpy as np

from tempus_copilot.rag.embed import FallbackEmbeddingClient, HashEmbeddingClient


def test_hash_embedding_client_is_deterministic() -> None:
    client = HashEmbeddingClient(dimension=8)
    vectors_a = client.embed_texts(["alpha", "beta"])
    vectors_b = client.embed_texts(["alpha", "beta"])
    assert vectors_a.shape == (2, 8)
    assert np.array_equal(vectors_a, vectors_b)


class FailingEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise httpx.HTTPError("transient failure")


def test_fallback_embedding_client_uses_fallback_on_http_error() -> None:
    fallback = HashEmbeddingClient(dimension=8)
    client = FallbackEmbeddingClient(primary=FailingEmbeddingClient(), fallback=fallback)
    vectors = client.embed_texts(["alpha"])
    assert vectors.shape == (1, 8)
