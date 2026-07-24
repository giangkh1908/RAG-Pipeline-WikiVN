"""Qdrant vector store for dense + sparse hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models

from rag_pipeline.config import QdrantConfig
from rag_pipeline.indexing.models import SearchResult
from rag_pipeline.storage.models import IndexEntry


@dataclass
class QdrantVectorStore:
    """Vector store backed by Qdrant with dense and sparse vector support."""

    config: QdrantConfig
    _client: QdrantClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = QdrantClient(url=self.config.url)

    def collection_exists(self) -> bool:
        """Return True if the collection already exists."""
        collections = self._client.get_collections().collections
        return any(c.name == self.config.collection_name for c in collections)

    def _existing_dense_dim(self) -> int | None:
        """Return the dense vector size of an existing collection.

        Returns ``None`` when the collection does not exist or the dense
        vector size cannot be determined. Used to detect a model switch that
        changed the embedding dimension so a stale-dim collection is not
        silently kept (which would crash at query time with a dim mismatch).
        """
        try:
            info = self._client.get_collection(self.config.collection_name)
        except Exception:
            return None
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            params = vectors.get(self.config.dense_vector_name)
            return params.size if params is not None else None
        return getattr(vectors, "size", None)

    def create_collection(self, dense_dim: int, recreate: bool = False) -> None:
        """Create the Qdrant collection with dense and sparse vectors.

        When the collection already exists and ``recreate`` is False, the
        existing dense dimension is compared to ``dense_dim`` and a
        ``RuntimeError`` is raised on mismatch — fail-fast at init time rather
        than silently keeping stale vectors that crash at query time. Pass
        ``recreate=True`` (or wipe the Qdrant volume) to re-index at a new dim.
        """
        if self.collection_exists():
            if not recreate:
                existing = self._existing_dense_dim()
                if existing is not None and existing != dense_dim:
                    raise RuntimeError(
                        f"Qdrant collection {self.config.collection_name!r} exists at "
                        f"dense dim {existing} but {dense_dim} was requested. Pass "
                        f"recreate=True or wipe the Qdrant volume to re-index."
                    )
                return
            self._client.delete_collection(self.config.collection_name)

        modifier = models.Modifier.IDF if self.config.sparse_modifier == "idf" else None
        self._client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config={
                self.config.dense_vector_name: models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self.config.sparse_vector_name: models.SparseVectorParams(
                    modifier=modifier,
                    index=models.SparseIndexParams(
                        on_disk=self.config.on_disk,
                    ),
                ),
            },
        )

    def upsert(self, entries: list[IndexEntry], batch_size: int = 100) -> None:
        """Upsert indexed entries into Qdrant in batches.

        Batching avoids Qdrant's default 32 MB payload limit when upserting
        many points with large dense vectors.
        """
        if not entries:
            return

        points: list[models.PointStruct] = []
        for entry in entries:
            if entry.dense_vector is None and entry.sparse_vector is None:
                continue

            vector: dict[str, Any] = {}
            if entry.dense_vector is not None:
                vector[self.config.dense_vector_name] = entry.dense_vector
            if entry.sparse_vector is not None:
                vector[self.config.sparse_vector_name] = models.SparseVector(
                    indices=list(entry.sparse_vector.keys()),
                    values=list(entry.sparse_vector.values()),
                )

            points.append(
                models.PointStruct(
                    id=str(entry.chunk_id),
                    vector=vector,
                    payload=entry.metadata,
                )
            )

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
            )

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search using dense vector."""
        top_k = top_k or self.config.dense_top_k
        search_filter = _build_filter(filters)

        results = self._client.search(
            collection_name=self.config.collection_name,
            query_vector=(self.config.dense_vector_name, query_vector),
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )

        return [_to_search_result(r) for r in results]

    def search_sparse(
        self,
        sparse_query: dict[int, float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search using BM25 sparse vector."""
        top_k = top_k or self.config.sparse_top_k
        search_filter = _build_filter(filters)

        sparse_vector = models.SparseVector(
            indices=list(sparse_query.keys()),
            values=list(sparse_query.values()),
        )

        results = self._client.search(
            collection_name=self.config.collection_name,
            query_vector=models.NamedSparseVector(
                name=self.config.sparse_vector_name,
                vector=sparse_vector,
            ),
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )

        return [_to_search_result(r) for r in results]


def _build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Build Qdrant filter from dict of payload key-value pairs."""
    if not filters:
        return None

    conditions = [
        models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in filters.items()
    ]
    return models.Filter(must=conditions)


def _to_search_result(point: Any) -> SearchResult:
    """Convert Qdrant search result to SearchResult."""
    return SearchResult(
        chunk_id=UUID(str(point.id)),
        score=float(point.score),
        metadata=point.payload or {},
    )
