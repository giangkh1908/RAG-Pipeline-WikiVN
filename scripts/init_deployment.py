"""Initialize a fresh deployment by ensuring Qdrant has indexed vectors.

This script is intended to run once when the stack starts (e.g. as a Docker
Compose init container). It behaves as follows:

1. If the Qdrant collection already contains points, do nothing.
2. Otherwise, if SQLite already has chunks, re-index those chunks into Qdrant.
3. Otherwise, run the full ingestion pipeline from the raw JSON dataset.

Usage:
    python scripts/init_deployment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_pipeline.config import DenseEmbeddingConfig, QdrantConfig, SparseEmbeddingConfig
from rag_pipeline.indexing import DenseEmbedder, IndexingService, QdrantVectorStore, SparseEmbedder
from rag_pipeline.storage import SQLiteStorage


def _qdrant_has_points(vector_store: QdrantVectorStore) -> bool:
    """Return True if the Qdrant collection exists and has points."""
    try:
        info = vector_store._client.get_collection(vector_store.config.collection_name)
        return info.points_count > 0
    except Exception:
        return False


def _sqlite_has_chunks(storage: SQLiteStorage) -> bool:
    """Return True if SQLite has at least one chunk."""
    for source in storage.list_sources():
        for doc in storage.list_documents(source_id=source.id):
            if storage.list_chunks(doc.id):
                return True
    return False


def _reindex_from_sqlite(storage: SQLiteStorage, vector_store: QdrantVectorStore) -> int:
    """Re-index all chunks currently stored in SQLite into Qdrant."""
    dense_embedder = DenseEmbedder(DenseEmbeddingConfig())
    sparse_embedder = SparseEmbedder(SparseEmbeddingConfig())
    indexing_service = IndexingService(
        storage=storage,
        vector_store=vector_store,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )

    total_indexed = 0
    for source in storage.list_sources():
        count = indexing_service.index_source(str(source.id))
        total_indexed += count
        print(f"  Source {source.id}: indexed {count} chunks")

    dense_embedder.close()
    return total_indexed


def _full_ingest(project_root: Path) -> int:
    """Run the full ingestion pipeline from the raw dataset."""
    from rag_pipeline.chunking import ChunkingPipeline, StructureChunker
    from rag_pipeline.config import RetrievalConfig, StorageConfig
    from rag_pipeline.indexing import IndexingService
    from rag_pipeline.ingestion import IngestionPipeline

    config = RetrievalConfig(
        qdrant=QdrantConfig(),
        dense=DenseEmbeddingConfig(),
        sparse=SparseEmbeddingConfig(),
        storage=StorageConfig(db_path=str(project_root / "data" / "rag_storage.db")),
    )
    storage = SQLiteStorage(config.storage.db_path)
    vector_store = QdrantVectorStore(config.qdrant)
    vector_store.create_collection(dense_dim=2048, recreate=False)

    dense_embedder = DenseEmbedder(config.dense)
    sparse_embedder = SparseEmbedder(config.sparse)
    indexing_service = IndexingService(
        storage=storage,
        vector_store=vector_store,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )

    pipeline = IngestionPipeline(
        storage=storage,
        chunking_pipeline=ChunkingPipeline(chunker=StructureChunker(max_tokens=200)),
        indexing_service=indexing_service,
    )

    dataset_path = project_root / "documents" / "vietnam_tourism_v2.json"
    source = pipeline.ingest_file(
        path=str(dataset_path),
        tenant_id="vietnam_tourism",
        source_type="vietnam_tourism",
        source_version="v2",
    )

    documents = storage.list_documents(source_id=source.id)
    total_chunks = sum(len(storage.list_chunks(doc.id)) for doc in documents)
    print(f"Full ingest complete: {len(documents)} documents, {total_chunks} chunks")

    dense_embedder.close()
    storage.close()
    return total_chunks


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    db_path = project_root / "data" / "rag_storage.db"
    storage = SQLiteStorage(str(db_path))
    vector_store = QdrantVectorStore(QdrantConfig())
    vector_store.create_collection(dense_dim=2048, recreate=False)

    print("=== Deployment Initialization ===")

    if _qdrant_has_points(vector_store):
        print("Qdrant already contains points. Skipping initialization.")
        storage.close()
        return 0

    if _sqlite_has_chunks(storage):
        print("SQLite has chunks but Qdrant is empty. Re-indexing from SQLite...")
        count = _reindex_from_sqlite(storage, vector_store)
        print(f"Re-indexed {count} chunks into Qdrant.")
    else:
        print("No local data found. Running full ingestion from JSON dataset...")
        count = _full_ingest(project_root)
        print(f"Ingested and indexed {count} chunks into Qdrant.")

    storage.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
