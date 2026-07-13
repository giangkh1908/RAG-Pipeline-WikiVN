"""Orchestration service for embedding chunks and indexing them in Qdrant."""

from __future__ import annotations

from typing import Any

from rag_pipeline.config import DenseEmbeddingConfig, QdrantConfig, SparseEmbeddingConfig
from rag_pipeline.indexing.embedders import DenseEmbedder, SparseEmbedder
from rag_pipeline.indexing.vector_store import QdrantVectorStore
from rag_pipeline.storage.base import Storage
from rag_pipeline.storage.models import Chunk, IndexEntry


class IndexingService:
    """Embed chunks and upsert them into Qdrant.

    The service reads chunks from the relational storage, generates dense and
    sparse vectors, persists the ``IndexEntry`` records back to storage, and
    finally upserts the vectors into Qdrant for hybrid retrieval.
    """

    def __init__(
        self,
        storage: Storage,
        vector_store: QdrantVectorStore | None = None,
        dense_embedder: DenseEmbedder | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        qdrant_config: QdrantConfig | None = None,
        dense_config: DenseEmbeddingConfig | None = None,
        sparse_config: SparseEmbeddingConfig | None = None,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store or QdrantVectorStore(qdrant_config or QdrantConfig())
        self.dense_embedder = dense_embedder or DenseEmbedder(
            dense_config or DenseEmbeddingConfig()
        )
        self.sparse_embedder = sparse_embedder or SparseEmbedder(
            sparse_config or SparseEmbeddingConfig()
        )

    def index_source(self, source_id: str) -> int:
        """Index all chunks belonging to a source. Returns number indexed."""
        from uuid import UUID

        documents = self.storage.list_documents(source_id=UUID(source_id))
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.storage.list_chunks(document.id))
        return self._index_chunks(chunks)

    def index_chunks(self, chunk_ids: list[str]) -> int:
        """Index a specific list of chunks by their IDs."""
        from uuid import UUID

        chunks = [self.storage.get_chunk(UUID(cid)) for cid in chunk_ids]
        chunks = [c for c in chunks if c is not None]
        return self._index_chunks(chunks)

    def _index_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        # Filter out chunks that are already indexed.
        chunks_to_index = [c for c in chunks if self.storage.get_index_entry(c.id) is None]
        if not chunks_to_index:
            return 0

        # Fit BM25 sparse embedder on the full corpus so that IDF and the
        # vocabulary are computed from all chunks being indexed.
        all_corpus_texts = [chunk.content for chunk in chunks]
        self.sparse_embedder.fit(all_corpus_texts)

        texts = [chunk.content for chunk in chunks_to_index]
        dense_vectors = self.dense_embedder.embed(texts)
        sparse_vectors = self.sparse_embedder.embed(texts)

        entries: list[IndexEntry] = []
        for chunk, dense, sparse in zip(chunks_to_index, dense_vectors, sparse_vectors):
            document = self.storage.get_document(chunk.document_id)
            payload: dict[str, Any] = {
                "document_id": str(chunk.document_id),
                "chunk_order": chunk.chunk_order,
            }
            if document is not None:
                payload["source_id"] = str(document.source_id)
                payload["title"] = document.metadata.get("title", "")
            payload["section_path"] = chunk.metadata.get("section_path", [])
            payload["is_reference_section"] = chunk.metadata.get("is_reference_section", False)

            entries.append(
                IndexEntry(
                    chunk_id=chunk.id,
                    dense_vector=dense,
                    sparse_vector=sparse,
                    metadata=payload,
                )
            )

        for entry in entries:
            self.storage.save_index_entry(entry)

        self.vector_store.upsert(entries)
        return len(entries)

    def close(self) -> None:
        self.dense_embedder.close()
