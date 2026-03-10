"""Embed text with the local Ollama embeddings API."""

from __future__ import annotations

from os import getenv
from time import sleep
from typing import Protocol, TypedDict, cast
from urllib.parse import urlparse

import httpx
import numpy as np


class EmbeddingClient(Protocol):
    """Protocol for embedding clients used by retrieval code."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed ordered texts into a float matrix.

        Args:
            texts: Ordered input texts to embed.

        Returns:
            A two-dimensional embedding matrix.
        """
        ...


class OllamaEmbedResponse(TypedDict, total=False):
    """Shape returned by the modern Ollama embeddings endpoint."""

    embeddings: list[list[float]]
    embedding: list[float]


class OllamaLegacyEmbedResponse(TypedDict, total=False):
    """Shape returned by the legacy Ollama embeddings endpoint."""

    embedding: list[float]


def _normalize_base_url(base_url: str) -> str:
    """Validate and normalize the local Ollama base URL.

    Args:
        base_url: Candidate base URL for the embeddings service.

    Returns:
        The normalized base URL without a trailing slash.

    Raises:
        ValueError: If the URL does not point to a local Ollama host.
    """
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    # Embeddings are intentionally local-only even though generation uses Ollama Cloud.
    if host not in {"localhost", "127.0.0.1"}:
        raise ValueError(
            "OLLAMA_EMBED_BASE_URL must point to local Ollama (localhost or 127.0.0.1)"
        )
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("OLLAMA_EMBED_BASE_URL must use http or https")
    return base_url.rstrip("/")


class OllamaEmbeddingClient:
    """Embedding client backed by a local Ollama server."""

    def __init__(self, model: str, request_retries: int = 2, backoff_seconds: float = 0.5) -> None:
        """Initialize the embedding client.

        Args:
            model: Embedding model name to request from Ollama.
            request_retries: Number of retry attempts after the initial request.
            backoff_seconds: Base backoff delay between retries.
        """
        base_url = getenv("OLLAMA_EMBED_BASE_URL") or "http://127.0.0.1:11434"
        self._base_url = _normalize_base_url(base_url)
        self._model = model
        self._request_retries = max(0, request_retries)
        self._backoff_seconds = max(0.0, backoff_seconds)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed one or more texts.

        Args:
            texts: Ordered input texts to embed.

        Returns:
            A 2D float32 matrix of embeddings.

        Raises:
            RuntimeError: If Ollama fails without producing a response.
            httpx.HTTPError: If all request attempts fail.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        response: httpx.Response | None = None
        for attempt in range(self._request_retries + 1):
            try:
                response = httpx.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": texts},
                    timeout=45.0,
                )
                # Older local Ollama builds still expose the legacy embeddings endpoint.
                if response.status_code == 404:
                    return self._embed_texts_legacy(texts)
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt >= self._request_retries:
                    raise
                sleep(self._backoff_seconds * (attempt + 1))
        if response is None:
            raise RuntimeError("Ollama embedding request failed without response")
        body = cast(OllamaEmbedResponse, response.json())
        embeddings = body.get("embeddings")
        if embeddings is None:
            single = body.get("embedding")
            embeddings = [single] if isinstance(single, list) else []
        return np.asarray(embeddings, dtype=np.float32)

    def _embed_texts_legacy(self, texts: list[str]) -> np.ndarray:
        """Embed texts with the legacy Ollama endpoint.

        Args:
            texts: Ordered input texts to embed.

        Returns:
            A 2D float32 matrix of embeddings.

        Raises:
            ValueError: If a legacy response omits the embedding vector.
            httpx.HTTPError: If the request fails.
        """
        # The legacy API is single-text per request, so compatibility is handled row by row.
        rows: list[list[float]] = []
        for text in texts:
            response = httpx.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=45.0,
            )
            response.raise_for_status()
            body = cast(OllamaLegacyEmbedResponse, response.json())
            vector = body.get("embedding")
            if not isinstance(vector, list):
                raise ValueError("Legacy embedding response missing embedding")
            rows.append([float(value) for value in vector])
        return np.asarray(rows, dtype=np.float32)
