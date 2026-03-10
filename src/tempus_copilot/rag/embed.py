from __future__ import annotations

from os import getenv
from time import sleep
from typing import Protocol, TypedDict, cast
from urllib.parse import urlparse

import httpx
import numpy as np


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


class OllamaEmbedResponse(TypedDict, total=False):
    embeddings: list[list[float]]
    embedding: list[float]


class OllamaLegacyEmbedResponse(TypedDict, total=False):
    embedding: list[float]


def _normalize_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host not in {"localhost", "127.0.0.1"}:
        raise ValueError(
            "OLLAMA_EMBED_BASE_URL must point to local Ollama (localhost or 127.0.0.1)"
        )
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("OLLAMA_EMBED_BASE_URL must use http or https")
    return base_url.rstrip("/")


class OllamaEmbeddingClient:
    def __init__(self, model: str, request_retries: int = 2, backoff_seconds: float = 0.5) -> None:
        base_url = getenv("OLLAMA_EMBED_BASE_URL") or "http://127.0.0.1:11434"
        self._base_url = _normalize_base_url(base_url)
        self._model = model
        self._request_retries = max(0, request_retries)
        self._backoff_seconds = max(0.0, backoff_seconds)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
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
