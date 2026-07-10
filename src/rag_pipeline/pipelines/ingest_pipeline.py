from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from rag_pipeline.indexing.bm25_index import BM25Index
from rag_pipeline.indexing.embedder import Embedder
from rag_pipeline.indexing.vector_store import VectorStore
from rag_pipeline.ingest.normalize import UVWWikipediaDocumentNormalizer
from rag_pipeline.models import CanonicalDocument, DocumentChunk, IndexingResult, IndexedChunk, SourceRecord
from rag_pipeline.transform.cleaner import WikipediaArticleCleaner
from rag_pipeline.transform.structure_chunker import StructuredChunker


class IngestPipeline:
    """End-to-end ingest flow: normalize → clean → chunk → BM25 insert → embed → index.

    Streaming batch with background flush: while the main thread reads,
    chunks, and inserts into BM25, a background thread embeds and upserts
    the previous batch to Qdrant.
    """

    def __init__(
        self,
        normalizer: UVWWikipediaDocumentNormalizer,
        cleaner: WikipediaArticleCleaner,
        chunker: StructuredChunker,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embed_batch_size: int = 500,
        flush_workers: int = 2,
        skip_qdrant_check: bool = False,
    ) -> None:
        self.normalizer = normalizer
        self.cleaner = cleaner
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embed_batch_size = embed_batch_size
        self.flush_workers = flush_workers
        self.skip_qdrant_check = skip_qdrant_check

    def run(self, records: Iterable[SourceRecord]) -> list[IndexingResult]:
        results: list[IndexingResult] = []
        seen: set[str] = set()

        # Streaming buffer: (text, doc_id, checksum, chunk, document)
        buffer: list[tuple[str, str, str, DocumentChunk, CanonicalDocument]] = []

        # Progress tracking
        start_time = time.perf_counter()
        total_docs = 0
        total_chunks = 0
        total_flushes = 0
        skipped = 0

        # Background flush pool — embed + upsert runs here
        executor = ThreadPoolExecutor(max_workers=self.flush_workers)
        pending_futures: list = []

        def _submit_flush():
            """Snapshot buffer, submit flush to background thread."""
            nonlocal total_flushes
            batch = buffer[: self.embed_batch_size]
            del buffer[: self.embed_batch_size]
            total_flushes += 1
            future = executor.submit(self._flush_batch, batch, results)
            pending_futures.append(future)

        def _wait_for_flushes():
            """Wait for all pending background flushes to complete."""
            for f in pending_futures:
                f.result()
            pending_futures.clear()

        for record in records:
            total_docs += 1
            document = self.normalizer.normalize(record)

            # Idempotent: skip already-indexed documents
            if document.checksum in seen:
                skipped += 1
                results.append(IndexingResult(document=document, chunks=[], updated=False))
                self._log_progress(total_docs, total_chunks, total_flushes, skipped, start_time)
                continue

            if not self.skip_qdrant_check and self.vector_store.has_document_version(document.doc_id, document.checksum):
                seen.add(document.checksum)
                skipped += 1
                results.append(IndexingResult(document=document, chunks=[], updated=False))
                self._log_progress(total_docs, total_chunks, total_flushes, skipped, start_time)
                continue

            seen.add(document.checksum)
            document.content = self.cleaner.clean(document.content)
            chunks = self.chunker.chunk(document)

            if not chunks:
                skipped += 1
                results.append(IndexingResult(document=document, chunks=[], updated=False))
                self._log_progress(total_docs, total_chunks, total_flushes, skipped, start_time)
                continue

            # BM25 insert — right after chunking, before embedding
            self._insert_bm25(chunks, document)

            for chunk in chunks:
                buffer.append((chunk.text, document.doc_id, document.checksum, chunk, document))
            total_chunks += len(chunks)

            # Flush full batches in background — main thread continues reading
            while len(buffer) >= self.embed_batch_size:
                _submit_flush()

            # Log every 500 docs
            if total_docs % 500 == 0:
                self._log_progress(total_docs, total_chunks, total_flushes, skipped, start_time)

        # Flush remaining buffer
        if buffer:
            _submit_flush()

        # Wait for all background flushes to finish
        _wait_for_flushes()
        executor.shutdown(wait=False)

        elapsed = time.perf_counter() - start_time
        indexed = sum(1 for r in results if r.updated)
        print(f"\n{'='*60}")
        print(f" DONE — {total_docs} docs | {indexed} indexed | {skipped} skipped")
        print(f" {total_chunks} chunks | {total_flushes} API batches | {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"{'='*60}")

        return results

    def _insert_bm25(self, chunks: list[DocumentChunk], document: CanonicalDocument) -> None:
        """Insert chunks into BM25 index (raw content only, not context prefix)."""
        items = []
        for chunk in chunks:
            _, raw_content = StructuredChunker.split_context_and_text(chunk.text)
            items.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": document.doc_id,
                "raw_content": raw_content,
                "full_text": chunk.text,
                "section_path": chunk.section_path,
                "checksum": document.checksum,
            })
        self.bm25_index.insert_batch(items)

    def _log_progress(self, total_docs: int, total_chunks: int, total_flushes: int, skipped: int, start_time: float) -> None:
        """Log progress to terminal."""
        if total_docs % 500 != 0:
            return

        elapsed = time.perf_counter() - start_time
        rate = total_docs / elapsed if elapsed > 0 else 0
        indexed = total_docs - skipped

        sys.stdout.write(f"\r\033[K")
        sys.stdout.write(
            f"📊 {total_docs:>8,} docs | "
            f"{indexed:>7,} indexed | "
            f"{skipped:>6,} skip | "
            f"{total_chunks:>9,} chunks | "
            f"{total_flushes:>4} batches | "
            f"{elapsed:>6.0f}s | "
            f"{rate:>7.0f} docs/s"
        )
        sys.stdout.flush()

    def _flush_batch(
        self,
        batch: list[tuple[str, str, str, DocumentChunk, CanonicalDocument]],
        results: list[IndexingResult],
    ) -> None:
        """Embed one batch and batch-upsert to Qdrant. Runs in background thread."""
        texts = [item[0] for item in batch]

        vectors = self.embedder.embed_texts(texts)

        # Group chunks by document
        doc_chunks: dict[str, tuple[str, str, list[tuple[DocumentChunk, list[float]]]]] = {}
        for i, (_, doc_id, checksum, chunk, _) in enumerate(batch):
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = (doc_id, checksum, [])
            doc_chunks[doc_id][2].append((chunk, vectors[i]))

        # Build all indexed chunks for batch upsert
        upsert_items: list[tuple[str, str, list[IndexedChunk]]] = []
        for doc_id, checksum, chunk_vec_pairs in doc_chunks.values():
            indexed_chunks = [
                IndexedChunk(chunk=c, dense_vector=v, sparse_vector=None)
                for c, v in chunk_vec_pairs
            ]
            upsert_items.append((doc_id, checksum, indexed_chunks))

        # Single batch upsert to Qdrant
        self.vector_store.upsert_batch(upsert_items)

        # Record results
        seen_in_batch: set[str] = set()
        for _, doc_id, checksum, chunk, document in batch:
            if doc_id not in seen_in_batch:
                seen_in_batch.add(doc_id)
                doc_chunks_list = [c for _, did, _, c, _ in batch if did == doc_id]
                results.append(IndexingResult(document=document, chunks=doc_chunks_list, updated=True))
