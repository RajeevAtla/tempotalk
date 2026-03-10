from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Protocol, TypedDict, cast

import numpy as np


class ScoredHit(TypedDict):
    metadata: dict[str, str]
    distance: float


class FaissIndexLike(Protocol):
    def add(self, vectors: np.ndarray) -> None: ...

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]: ...


class FaissModule(Protocol):
    def IndexFlatL2(self, dimension: int) -> FaissIndexLike: ...


def _load_faiss() -> FaissModule:
    return cast(FaissModule, import_module("faiss"))


class FaissIndex:
    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        faiss = _load_faiss()
        index_ctor = getattr(faiss, "IndexFlatL2", None)
        if index_ctor is None:
            raise RuntimeError("faiss.IndexFlatL2 is unavailable in this environment")
        self._index = index_ctor(dimension)
        self._metadata: list[dict[str, str]] = []

    def add(self, vectors: np.ndarray, metadata: Sequence[Mapping[str, str]]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        if vectors.shape[0] != len(metadata):
            raise ValueError("vector count and metadata length must match")
        self._index.add(vectors)
        self._metadata.extend(dict(item) for item in metadata)

    def query(self, vector: np.ndarray, top_k: int) -> list[dict[str, str]]:
        return [item["metadata"] for item in self.query_with_scores(vector=vector, top_k=top_k)]

    def query_with_scores(self, vector: np.ndarray, top_k: int) -> list[ScoredHit]:
        if top_k <= 0:
            return []
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        if vector.ndim == 1:
            query = vector.reshape(1, -1)
        else:
            query = vector
        distances, indices = self._index.search(query, top_k)
        hits: list[ScoredHit] = []
        for pos, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            hits.append(
                {
                    "metadata": self._metadata[int(idx)],
                    "distance": float(distances[0][pos]),
                }
            )
        return hits
