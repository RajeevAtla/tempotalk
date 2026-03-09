from __future__ import annotations

from typing import Any

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        index_ctor = getattr(faiss, "IndexFlatL2", None)
        if index_ctor is None:
            raise RuntimeError("faiss.IndexFlatL2 is unavailable in this environment")
        self._index: Any = index_ctor(dimension)
        self._metadata: list[dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        if vectors.shape[0] != len(metadata):
            raise ValueError("vector count and metadata length must match")
        self._index.add(vectors)
        self._metadata.extend(metadata)

    def query(self, vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        return [item["metadata"] for item in self.query_with_scores(vector=vector, top_k=top_k)]

    def query_with_scores(self, vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        if vector.ndim == 1:
            query = vector.reshape(1, -1)
        else:
            query = vector
        distances, indices = self._index.search(query, top_k)
        hits: list[dict[str, Any]] = []
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
