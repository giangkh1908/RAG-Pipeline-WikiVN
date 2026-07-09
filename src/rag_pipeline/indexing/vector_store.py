from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_pipeline.config import QdrantConfig
from rag_pipeline.models import DocumentChunk, IndexedChunk


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


@dataclass(slots=True)
class SearchResult:
    """A single search result from vector store."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    def has_document_version(self, doc_id: str, checksum: str) -> bool:
        """Return True when the exact document version is already indexed."""

    def upsert_document(self, doc_id: str, checksum: str, chunks: list[IndexedChunk]) -> None:
        """Persist a document version and its chunk vectors."""

    def upsert_batch(self, items: list[tuple[str, str, list[IndexedChunk]]]) -> None:
        """Batch upsert multiple documents in one call. items = [(doc_id, checksum, chunks), ...]"""

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks. Returns top_k results sorted by score."""


@dataclass(slots=True)
class InMemoryVectorStore:
    documents: dict[str, str] = field(default_factory=dict)
    chunks: dict[str, IndexedChunk] = field(default_factory=dict)

    def has_document_version(self, doc_id: str, checksum: str) -> bool:
        return self.documents.get(doc_id) == checksum

    def upsert_document(self, doc_id: str, checksum: str, chunks: list[IndexedChunk]) -> None:
        self.documents[doc_id] = checksum
        for chunk in chunks:
            self.chunks[chunk.chunk.chunk_id] = chunk

    def upsert_batch(self, items: list[tuple[str, str, list[IndexedChunk]]]) -> None:
        for doc_id, checksum, chunks in items:
            self.upsert_document(doc_id, checksum, chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search using cosine similarity (in-memory)."""

        scored: list[SearchResult] = []
        for chunk_id, indexed in self.chunks.items():
            # Apply filters if provided
            if filters:
                skip = False
                for key, value in filters.items():
                    if indexed.chunk.metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            score = _cosine_sim(query_vector, indexed.dense_vector)
            scored.append(SearchResult(
                chunk_id=chunk_id,
                doc_id=indexed.chunk.doc_id,
                text=indexed.chunk.text,
                score=score,
                metadata=indexed.chunk.metadata,
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


def _hex_to_uuid(hex_str: str) -> str:
    """Pad a 24-char hex string to full UUID format (32 hex + 4 hyphens)."""
    padded = hex_str.ljust(32, "0")
    return f"{padded[:8]}-{padded[8:12]}-{padded[12:16]}-{padded[16:20]}-{padded[20:32]}"


def _id_from_str(value: str) -> str:
    """Chunk ID is stored as a deterministic UUID derived from the original hex string."""
    return _hex_to_uuid(value)


@dataclass
class QdrantVectorStore:
    config: QdrantConfig
    vector_size: int = 0  # 0 = use config.vector_size
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.vector_size == 0:
            self.vector_size = self.config.vector_size
        self._client = self._get_client()
        self._ensure_collection()

    def _get_client(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "The `qdrant-client` package is required for Qdrant persistence. "
                "Install with `pip install .[indexing]`."
            ) from exc
        return QdrantClient(url=self.config.url)

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        collections = self._client.get_collections().collections
        names = [c.name for c in collections]
        if self.config.collection_name not in names:
            self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config={
                    self.config.dense_vector_name: VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                },
            )

    def has_document_version(self, doc_id: str, checksum: str) -> bool:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        scroll_filter = Filter(
            must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="doc_checksum", match=MatchValue(value=checksum)),
            ]
        )
        points, _ = self._client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=scroll_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return bool(points)

    def upsert_document(self, doc_id: str, checksum: str, chunks: list[IndexedChunk]) -> None:
        points = self._build_points(doc_id, checksum, chunks)
        self._client.upsert(collection_name=self.config.collection_name, points=points)

    def upsert_batch(self, items: list[tuple[str, str, list[IndexedChunk]]]) -> None:
        """Batch upsert all points from multiple documents in one Qdrant call."""
        all_points: list = []
        for doc_id, checksum, chunks in items:
            all_points.extend(self._build_points(doc_id, checksum, chunks))

        if all_points:
            self._client.upsert(collection_name=self.config.collection_name, points=all_points)

    def _build_points(self, doc_id: str, checksum: str, chunks: list[IndexedChunk]) -> list:
        from qdrant_client.models import PointStruct

        points = []
        for indexed in chunks:
            chunk: DocumentChunk = indexed.chunk
            payload = {
                "doc_id": doc_id,
                "doc_checksum": checksum,
                "text": chunk.text,
                "section_path": chunk.section_path,
                "article_number": chunk.article_number,
                "clause_number": chunk.clause_number,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "parent_chunk_id": chunk.parent_chunk_id,
                "prev_chunk_id": chunk.prev_chunk_id,
                "next_chunk_id": chunk.next_chunk_id,
                **chunk.metadata,
            }
            vector_payload = {self.config.dense_vector_name: indexed.dense_vector}
            if indexed.sparse_vector:
                vector_payload[self.config.sparse_vector_name] = indexed.sparse_vector
            point_id = _id_from_str(chunk.chunk_id)
            points.append(PointStruct(id=point_id, vector=vector_payload, payload=payload))

        return points

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search Qdrant for similar chunks using dense vector."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        search_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            search_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self.config.collection_name,
            query_vector=(self.config.dense_vector_name, query_vector),
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )

        return [
            SearchResult(
                chunk_id=str(point.id),
                doc_id=point.payload.get("doc_id", ""),
                text=point.payload.get("text", ""),
                score=point.score,
                metadata=point.payload,
            )
            for point in results
        ]
