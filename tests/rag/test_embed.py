import httpx
import numpy as np
import pytest

from tempus_copilot.rag import embed as embed_module
from tempus_copilot.rag.embed import (
    FallbackEmbeddingClient,
    GeminiEmbeddingClient,
    HashEmbeddingClient,
)


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


def test_hash_embedding_client_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError):
        HashEmbeddingClient(dimension=0)


def test_gemini_embedding_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        GeminiEmbeddingClient(model="gemini-embedding-001")


def test_gemini_embedding_client_handles_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    client = GeminiEmbeddingClient(model="gemini-embedding-001")
    out = client.embed_texts([])
    assert out.shape == (0, 0)


def test_gemini_embedding_client_success_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    calls: dict[str, int] = {"count": 0}

    def fake_post(*_: object, **__: object) -> FakeResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPError("temporary")
        return FakeResponse({"embeddings": [{"values": [1.0, 2.0]}, {"values": [3.0, 4.0]}]})

    monkeypatch.setattr(embed_module.httpx, "post", fake_post)
    monkeypatch.setattr(embed_module, "sleep", lambda _: None)
    client = GeminiEmbeddingClient(model="gemini-embedding-001", request_retries=1)
    out = client.embed_texts(["a", "b"])
    assert calls["count"] == 2
    assert out.shape == (2, 2)


def test_gemini_embedding_client_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setattr(
        embed_module.httpx,
        "post",
        lambda *_, **__: (_ for _ in ()).throw(httpx.HTTPError("nope")),
    )
    monkeypatch.setattr(embed_module, "sleep", lambda _: None)
    client = GeminiEmbeddingClient(model="gemini-embedding-001", request_retries=0)
    with pytest.raises(httpx.HTTPError):
        client.embed_texts(["a"])


def test_gemini_embedding_client_no_attempts_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    client = GeminiEmbeddingClient(model="gemini-embedding-001", request_retries=-1)
    with pytest.raises(RuntimeError):
        client.embed_texts(["a"])
