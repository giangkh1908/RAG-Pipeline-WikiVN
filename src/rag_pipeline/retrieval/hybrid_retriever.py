"""Hybrid dense + sparse retriever with RRF fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import UUID

from rag_pipeline.config import RetrievalConfig
from rag_pipeline.indexing import DenseEmbedder, QdrantVectorStore, SparseEmbedder
from rag_pipeline.indexing.models import SearchResult
from rag_pipeline.retrieval.models import RetrievalResult
from rag_pipeline.storage.base import Storage


class HybridRetriever:
    """Retrieve chunks using dense + sparse hybrid search and RRF fusion."""

    def __init__(
        self,
        config: RetrievalConfig,
        storage: Storage,
        vector_store: QdrantVectorStore,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
    ) -> None:
        self.config = config
        self.storage = storage
        self.vector_store = vector_store
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder

    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize a raw query string."""
        return " ".join(query.lower().split())

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve and rank chunks for a query.

        Parameters
        ----------
        query:
            Raw user query.
        top_k:
            Number of final results. Defaults to ``config.rrf_top_k``.
        filters:
            Optional Qdrant payload filters (key-value pairs).

        Returns
        -------
        Ranked list of ``RetrievalResult``.
        """
        top_k = top_k or self.config.rrf_top_k
        normalized_query = self.normalize_query(query)

        dense_vector = self.dense_embedder.embed([normalized_query])[0]
        sparse_vector = self.sparse_embedder.embed([normalized_query])[0]

        dense_results = self.vector_store.search_dense(
            query_vector=dense_vector,
            top_k=self.config.qdrant.dense_top_k,
            filters=filters,
        )
        sparse_results = self.vector_store.search_sparse(
            sparse_query=sparse_vector,
            top_k=self.config.qdrant.sparse_top_k,
            filters=filters,
        )

        ranked = self._rrf_fuse(dense_results, sparse_results, top_k=top_k)
        return self._build_results(ranked, dense_results, sparse_results)

    def _rrf_fuse(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        top_k: int,
    ) -> list[tuple[UUID, float]]:
        """Fuse dense and sparse rankings using Reciprocal Rank Fusion."""
        scores: dict[UUID, float] = defaultdict(float)

        for rank, result in enumerate(dense_results):
            scores[result.chunk_id] += 1.0 / (self.config.rrf_k + rank + 1)

        for rank, result in enumerate(sparse_results):
            scores[result.chunk_id] += 1.0 / (self.config.rrf_k + rank + 1)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    def _build_results(
        self,
        ranked: list[tuple[UUID, float]],
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
    ) -> list[RetrievalResult]:
        """Build ``RetrievalResult`` objects with chunk content and original scores."""
        dense_by_id = {r.chunk_id: r for r in dense_results}
        sparse_by_id = {r.chunk_id: r for r in sparse_results}

        results: list[RetrievalResult] = []
        for rank, (chunk_id, rrf_score) in enumerate(ranked, start=1):
            chunk = self.storage.get_chunk(chunk_id)
            content = chunk.content if chunk is not None else ""

            dense_result = dense_by_id.get(chunk_id)
            sparse_result = sparse_by_id.get(chunk_id)
            metadata = dense_result.metadata if dense_result else {}
            if sparse_result and not metadata:
                metadata = sparse_result.metadata

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=content,
                    rrf_score=rrf_score,
                    rank=rank,
                    dense_score=dense_result.score if dense_result else None,
                    sparse_score=sparse_result.score if sparse_result else None,
                    metadata=metadata,
                )
            )
        return results
