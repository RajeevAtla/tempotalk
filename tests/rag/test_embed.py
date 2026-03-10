import httpx
import numpy as np
import pytest

from tempus_copilot.rag import embed as embed_module
from tempus_copilot.rag.embed import OllamaEmbeddingClient, _normalize_base_url


def test_ollama_embedding_client_rejects_non_local_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "https://ollama.com")
    with pytest.raises(ValueError):
        OllamaEmbeddingClient(model="embeddinggemma")


def test_normalize_base_url_accepts_localhost_and_trims_trailing_slash() -> None:
    assert _normalize_base_url("http://localhost:11434/") == "http://localhost:11434"


def test_normalize_base_url_rejects_invalid_scheme() -> None:
    with pytest.raises(ValueError):
        _normalize_base_url("ftp://127.0.0.1:11434")


def test_ollama_embedding_client_handles_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://127.0.0.1:11434")
    client = OllamaEmbeddingClient(model="embeddinggemma")
    out = client.embed_texts([])
    assert out.shape == (0, 0)


def test_ollama_embedding_client_success_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://127.0.0.1:11434")

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    calls = {"count": 0}

    def fake_post(*_: object, **__: object) -> FakeResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPError("temporary")
        return FakeResponse({"embeddings": [[1.0, 2.0], [3.0, 4.0]]})

    monkeypatch.setattr(embed_module.httpx, "post", fake_post)
    monkeypatch.setattr(embed_module, "sleep", lambda _: None)
    client = OllamaEmbeddingClient(model="embeddinggemma", request_retries=1)
    out = client.embed_texts(["a", "b"])
    assert calls["count"] == 2
    assert out.shape == (2, 2)
    assert np.allclose(out, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_ollama_embedding_client_accepts_single_embedding_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://127.0.0.1:11434")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"embedding": [1.5, 2.5, 3.5]}

    monkeypatch.setattr(embed_module.httpx, "post", lambda *_, **__: FakeResponse())
    client = OllamaEmbeddingClient(model="embeddinggemma")
    out = client.embed_texts(["a"])
    assert out.shape == (1, 3)
    assert np.allclose(out, np.array([[1.5, 2.5, 3.5]], dtype=np.float32))


def test_ollama_embedding_client_returns_empty_array_when_payload_has_no_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://127.0.0.1:11434")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"ignored": "value"}

    monkeypatch.setattr(embed_module.httpx, "post", lambda *_, **__: FakeResponse())
    client = OllamaEmbeddingClient(model="embeddinggemma")
    out = client.embed_texts(["a"])
    assert out.dtype == np.float32
    assert out.shape == (0,)


def test_ollama_embedding_client_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(
        embed_module.httpx,
        "post",
        lambda *_, **__: (_ for _ in ()).throw(httpx.HTTPError("nope")),
    )
    monkeypatch.setattr(embed_module, "sleep", lambda _: None)
    client = OllamaEmbeddingClient(model="embeddinggemma", request_retries=0)
    with pytest.raises(httpx.HTTPError):
        client.embed_texts(["a"])


def test_ollama_embedding_client_runtime_error_when_retry_loop_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(embed_module, "range", lambda count: [], raising=False)
    client = OllamaEmbeddingClient(model="embeddinggemma")
    with pytest.raises(RuntimeError, match="without response"):
        client.embed_texts(["a"])


def test_ollama_embedding_client_falls_back_to_legacy_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://127.0.0.1:11434")

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "error",
                    request=httpx.Request("POST", "http://127.0.0.1"),
                    response=httpx.Response(self.status_code),
                )

        def json(self) -> dict[str, object]:
            return self._payload

    calls: list[str] = []

    def fake_post(url: str, *_: object, **__: object) -> FakeResponse:
        calls.append(url)
        if url.endswith("/api/embed"):
            return FakeResponse(404, {"error": "not found"})
        return FakeResponse(200, {"embedding": [0.1, 0.2, 0.3]})

    monkeypatch.setattr(embed_module.httpx, "post", fake_post)
    client = OllamaEmbeddingClient(model="embeddinggemma")
    out = client.embed_texts(["a", "b"])
    assert out.shape == (2, 3)
    assert any(url.endswith("/api/embeddings") for url in calls)


def test_ollama_embedding_client_legacy_endpoint_requires_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://127.0.0.1:11434")

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "error",
                    request=httpx.Request("POST", "http://127.0.0.1"),
                    response=httpx.Response(self.status_code),
                )

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_post(url: str, *_: object, **__: object) -> FakeResponse:
        if url.endswith("/api/embed"):
            return FakeResponse(404, {"error": "not found"})
        return FakeResponse(200, {"ignored": True})

    monkeypatch.setattr(embed_module.httpx, "post", fake_post)
    client = OllamaEmbeddingClient(model="embeddinggemma")
    with pytest.raises(ValueError):
        client.embed_texts(["a"])


def test_ollama_embedding_client_legacy_endpoint_coerces_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_EMBED_BASE_URL", "http://127.0.0.1:11434")

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "error",
                    request=httpx.Request("POST", "http://127.0.0.1"),
                    response=httpx.Response(self.status_code),
                )

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_post(url: str, *_: object, **__: object) -> FakeResponse:
        if url.endswith("/api/embed"):
            return FakeResponse(404, {"error": "not found"})
        return FakeResponse(200, {"embedding": [1, "2.5", 3.0]})

    monkeypatch.setattr(embed_module.httpx, "post", fake_post)
    client = OllamaEmbeddingClient(model="embeddinggemma")
    out = client.embed_texts(["a"])
    assert out.dtype == np.float32
    assert np.allclose(out, np.array([[1.0, 2.5, 3.0]], dtype=np.float32))
