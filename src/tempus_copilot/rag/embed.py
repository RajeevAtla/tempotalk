from __future__ import annotations

from hashlib import sha256
from os import getenv
from time import sleep
from typing import Protocol

import httpx
import numpy as np


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


class HashEmbeddingClient:
    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            digest = sha256(text.encode("utf-8")).digest()
            values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            tiled = np.resize(values, self._dimension)
            norm = np.linalg.norm(tiled)
            rows.append(tiled if norm == 0 else tiled / norm)
        return np.stack(rows).astype(np.float32)


class GeminiEmbeddingClient:
    def __init__(self, model: str, request_retries: int = 2, backoff_seconds: float = 0.5) -> None:
        api_key = getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini embeddings")
        self._api_key = api_key
        self._model = model
        self._request_retries = request_retries
        self._backoff_seconds = backoff_seconds
        self._url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:batchEmbedContents"
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        requests = [
            {
                "content": {
                    "parts": [{"text": text}],
                }
            }
            for text in texts
        ]
        payload = {"requests": requests}
        response: httpx.Response | None = None
        for attempt in range(self._request_retries + 1):
            try:
                response = httpx.post(
                    self._url,
                    params={"key": self._api_key},
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt >= self._request_retries:
                    raise
                sleep(self._backoff_seconds * (attempt + 1))
        if response is None:
            raise RuntimeError("Gemini embedding request failed without response")
        data = response.json()
        embeds = data.get("embeddings", [])
        matrix = [item.get("values", []) for item in embeds]
        return np.asarray(matrix, dtype=np.float32)


class FallbackEmbeddingClient:
    def __init__(self, primary: EmbeddingClient, fallback: EmbeddingClient) -> None:
        self._primary = primary
        self._fallback = fallback

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        try:
            return self._primary.embed_texts(texts)
        except httpx.HTTPError:
            return self._fallback.embed_texts(texts)
