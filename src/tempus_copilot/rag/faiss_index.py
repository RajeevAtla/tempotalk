"""Thin typed wrapper around a FAISS flat L2 index."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Protocol, TypedDict, cast

import numpy as np


class ScoredHit(TypedDict):
    """Search hit with attached metadata and distance score."""

    metadata: dict[str, str]
    distance: float


class FaissIndexLike(Protocol):
    """Minimal FAISS index surface required by the wrapper."""

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the backing FAISS index.

        Args:
            vectors: Two-dimensional vector matrix.
        """
        ...

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the backing FAISS index.

        Args:
            vectors: Two-dimensional query matrix.
            top_k: Maximum number of hits to return.

        Returns:
            Distances and indices from the FAISS search result.
        """
        ...


class FaissModule(Protocol):
    """Typed view of the FAISS module entry points used here."""

    def IndexFlatL2(self, dimension: int) -> FaissIndexLike:
        """Create an L2 flat index.

        Args:
            dimension: Embedding vector width.

        Returns:
            A FAISS index compatible with the local wrapper protocol.
        """
        ...


def _load_faiss() -> FaissModule:
    """Import the FAISS module lazily.

    Returns:
        The imported FAISS module cast to the protocol used locally.
    """
    return cast(FaissModule, import_module("faiss"))


class FaissIndex:
    """Typed convenience wrapper over ``faiss.IndexFlatL2``."""

    def __init__(self, dimension: int) -> None:
        """Create an empty index.

        Args:
            dimension: Embedding vector width.

        Raises:
            ValueError: If the dimension is not positive.
            RuntimeError: If the required FAISS constructor is unavailable.
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        faiss = _load_faiss()
        index_ctor = getattr(faiss, "IndexFlatL2", None)
        if index_ctor is None:
            raise RuntimeError("faiss.IndexFlatL2 is unavailable in this environment")
        self._index = index_ctor(dimension)
        self._metadata: list[dict[str, str]] = []

    def add(self, vectors: np.ndarray, metadata: Sequence[Mapping[str, str]]) -> None:
        """Add vectors and aligned metadata rows to the index.

        Args:
            vectors: 2D embedding matrix.
            metadata: Metadata rows aligned one-to-one with the vectors.

        Raises:
            ValueError: If the vectors are not 2D or do not align with metadata.
        """
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        if vectors.shape[0] != len(metadata):
            raise ValueError("vector count and metadata length must match")
        self._index.add(vectors)
        self._metadata.extend(dict(item) for item in metadata)

    def query(self, vector: np.ndarray, top_k: int) -> list[dict[str, str]]:
        """Query the index and return only metadata rows.

        Args:
            vector: Query embedding vector.
            top_k: Maximum number of hits to return.

        Returns:
            Metadata rows for the matching hits.
        """
        return [item["metadata"] for item in self.query_with_scores(vector=vector, top_k=top_k)]

    def query_with_scores(self, vector: np.ndarray, top_k: int) -> list[ScoredHit]:
        """Query the index and return metadata with L2 distances.

        Args:
            vector: Query embedding vector or batch containing one vector.
            top_k: Maximum number of hits to return.

        Returns:
            Search hits ordered by FAISS result position.
        """
        if top_k <= 0:
            return []
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        # FAISS expects a batch dimension even for a single query vector.
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
