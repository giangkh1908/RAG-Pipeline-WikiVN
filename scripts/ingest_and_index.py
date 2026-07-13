"""Full ingestion + embedding + indexing pipeline for Vietnam Tourism dataset.

This script:
    1. Loads environment variables from `.env`.
    2. Parses `documents/vietnam_tourism_v2.json` into chunks.
    3. Stores chunks in SQLite.
    4. Generates dense (OpenRouter) and sparse (BM25) vectors.
    5. Upserts vectors into Qdrant for hybrid retrieval.

Example:
    python scripts/ingest_and_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_pipeline.chunking import ChunkingPipeline, StructureChunker
from rag_pipeline.config import (
    DenseEmbeddingConfig,
    QdrantConfig,
    RetrievalConfig,
    SparseEmbeddingConfig,
    StorageConfig,
)
from rag_pipeline.indexing import DenseEmbedder, IndexingService, QdrantVectorStore, SparseEmbedder
from rag_pipeline.ingestion import IngestionPipeline
from rag_pipeline.storage import SQLiteStorage


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    config = RetrievalConfig(
        qdrant=QdrantConfig(),
        dense=DenseEmbeddingConfig(),
        sparse=SparseEmbeddingConfig(),
        storage=StorageConfig(db_path=str(project_root / "data" / "rag_storage.db")),
    )

    db_path = Path(config.storage.db_path)
    dataset_path = project_root / "documents" / "vietnam_tourism_v2.json"

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    storage = SQLiteStorage(str(db_path))

    print("=== Ingestion + Indexing Pipeline ===")
    print(f"Dataset : {dataset_path}")
    print(f"SQLite  : {db_path}")
    print(f"Qdrant  : {config.qdrant.url}")
    print(f"Dense   : {config.dense.model_name}")
    print(f"Sparse  : {config.sparse.model_name}")
    print()

    # Prepare vector store and create collection up-front.
    vector_store = QdrantVectorStore(config.qdrant)
    print("Creating Qdrant collection if needed...")
    vector_store.create_collection(dense_dim=2048, recreate=False)

    # Prepare embedders and indexing service.
    dense_embedder = DenseEmbedder(config.dense)
    sparse_embedder = SparseEmbedder(config.sparse)
    indexing_service = IndexingService(
        storage=storage,
        vector_store=vector_store,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )

    # Run ingestion; chunking is followed by embedding + Qdrant upsert.
    pipeline = IngestionPipeline(
        storage=storage,
        chunking_pipeline=ChunkingPipeline(chunker=StructureChunker(max_tokens=200)),
        indexing_service=indexing_service,
    )

    print("Ingesting and indexing...")
    source = pipeline.ingest_file(
        path=str(dataset_path),
        tenant_id="vietnam_tourism",
        source_type="vietnam_tourism",
        source_version="v2",
    )

    # Summary from SQLite.
    documents = storage.list_documents(source_id=source.id)
    total_chunks = sum(len(storage.list_chunks(doc.id)) for doc in documents)
    index_entries = storage.list_index_entries()

    print()
    print("=== Summary ===")
    print(f"Source id      : {source.id}")
    print(f"Documents      : {len(documents)}")
    print(f"Chunks         : {total_chunks}")
    print(f"Index entries  : {len(index_entries)}")

    # Summary from Qdrant.
    try:
        collection_info = vector_store._client.get_collection(config.qdrant.collection_name)
        print(f"Qdrant points  : {collection_info.points_count}")
    except Exception as exc:
        print(f"Qdrant info    : unavailable ({exc})")

    dense_embedder.close()
    storage.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
