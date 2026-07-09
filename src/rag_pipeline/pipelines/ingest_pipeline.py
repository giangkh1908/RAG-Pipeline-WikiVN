from __future__ import annotations

import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag_pipeline.indexing.bm25_index import BM25Index
from rag_pipeline.indexing.embedder import Embedder
from rag_pipeline.indexing.vector_store import VectorStore
from rag_pipeline.ingest.normalize import UVWWikipediaDocumentNormalizer
from rag_pipeline.models import CanonicalDocument, DocumentChunk, IndexingResult, IndexedChunk, SourceRecord
from rag_pipeline.transform.cleaner import WikipediaArticleCleaner
from rag_pipeline.transform.structure_chunker import StructuredChunker


class IngestPipeline:
    """2-phase ingest pipeline: chunking (offline) → embed + index (online).

    Phase 1 (run_chunking): raw docs → normalize → clean → chunk → JSONL output
    Phase 2 (run_embedding): chunks JSONL → embed → Qdrant + BM25 index

    Streaming batch with background flush: while the main thread reads and
    chunks documents, a background thread embeds and upserts the previous batch.
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

        # BM25 corpus: (chunk_id, raw_text) pairs collected for final index build
        bm25_docs: list[tuple[str, str]] = []

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

            for chunk in chunks:
                buffer.append((chunk.text, document.doc_id, document.checksum, chunk, document))
                bm25_docs.append((chunk.chunk_id, self._raw_chunk_text(chunk)))
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

        # Build BM25 keyword index from all ingested chunks
        if bm25_docs:
            self.bm25_index.build(bm25_docs)

        elapsed = time.perf_counter() - start_time
        indexed = sum(1 for r in results if r.updated)
        print(f"\n{'='*60}")
        print(f" DONE — {total_docs} docs | {indexed} indexed | {skipped} skipped")
        print(f" {total_chunks} chunks | {total_flushes} API batches | {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"{'='*60}")

        return results

    def run_chunking(self, records: Iterable[SourceRecord], output_path: Path) -> int:
        """Phase 1: chunk documents → JSONL output.

        Args:
            records: Raw document records to chunk.
            output_path: Path to write chunked JSONL file.

        Returns:
            Total number of chunks written.
        """
        import gzip
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_docs = 0
        total_chunks = 0
        skipped = 0
        start_time = time.perf_counter()

        # Choose plain or gzip output
        if output_path.suffix == ".gz":
            fout_ctx = gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6)
        else:
            fout_ctx = open(output_path, "w", encoding="utf-8")

        with fout_ctx as fout:
            for record in records:
                total_docs += 1
                document = self.normalizer.normalize(record)

                # Skip Wikipedia module / template pages
                if document.title.startswith(("Mô đun:", "Module:")):
                    skipped += 1
                    continue

                document.content = self.cleaner.clean(document.content)
                chunks = self.chunker.chunk(document)

                if not chunks:
                    skipped += 1
                    continue

                for chunk in chunks:
                    context, text = StructuredChunker.split_context_and_text(chunk.text)
                    out = {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "title": document.title,
                        "source_url": document.source_url,
                        "section_path": chunk.section_path,
                        "context": context,
                        "text": text,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "is_reference_section": chunk.metadata.get("is_reference_section", False),
                        "prev_chunk_id": chunk.prev_chunk_id,
                        "next_chunk_id": chunk.next_chunk_id,
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    total_chunks += 1

                # Log every 1000 docs
                if total_docs % 1000 == 0:
                    self._log_chunking_progress(total_docs, total_chunks, skipped, start_time)

        elapsed = time.perf_counter() - start_time
        print(f"\n{'='*60}")
        print(f" CHUNK DONE — {total_docs:,} docs | {total_chunks:,} chunks | {skipped:,} skipped")
        print(f" Output: {output_path.resolve()}")
        print(f" Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"{'='*60}")

        return total_chunks

    def run_embedding(self, chunk_path: Path, use_qdrant: bool = True) -> list[IndexingResult]:
        """Phase 2: chunks JSONL → embed → Qdrant + BM25.

        Args:
            chunk_path: Path to pre-chunked JSONL file.
            use_qdrant: Whether to use Qdrant (True) or InMemory (False).

        Returns:
            List of IndexingResult for each document.
        """
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        from rag_pipeline.ingest.dataset import ChunkedJsonlReader

        reader = ChunkedJsonlReader(chunk_path)
        results: list[IndexingResult] = []
        seen: set[str] = set()

        # Streaming buffer: (text, doc_id, checksum, chunk, document)
        buffer: list[tuple[str, str, str, DocumentChunk, CanonicalDocument]] = []

        # BM25 corpus: (chunk_id, raw_text) pairs
        bm25_docs: list[tuple[str, str]] = []

        # Progress tracking
        start_time = time.perf_counter()
        total_chunks = 0
        total_flushes = 0
        skipped = 0

        # Background flush pool
        executor = ThreadPoolExecutor(max_workers=self.flush_workers)
        pending_futures: list = []

        def _submit_flush():
            nonlocal total_flushes
            batch = buffer[: self.embed_batch_size]
            del buffer[: self.embed_batch_size]
            total_flushes += 1
            future = executor.submit(self._flush_batch, batch, results)
            pending_futures.append(future)

        def _wait_for_flushes():
            for f in pending_futures:
                f.result()
            pending_futures.clear()

        for record in reader.read():
            payload = record.payload
            chunk_id = payload.get("chunk_id")
            doc_id = payload.get("doc_id", "")
            text = payload.get("text", "")
            context = payload.get("context", "")

            # Reconstruct full chunk text with context prefix
            full_text = f"{context}\n\n{text}" if context else text

            # Create DocumentChunk
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=full_text,
                section_path=payload.get("section_path", []),
                article_number=payload.get("article_number"),
                clause_number=payload.get("clause_number"),
                chunk_index=payload.get("chunk_index", 0),
                token_count=payload.get("token_count", 0),
                parent_chunk_id=payload.get("parent_chunk_id"),
                prev_chunk_id=payload.get("prev_chunk_id"),
                next_chunk_id=payload.get("next_chunk_id"),
                checksum=chunk_id,
                metadata={
                    "title": payload.get("title", ""),
                    "source_url": payload.get("source_url", ""),
                    "is_reference_section": payload.get("is_reference_section", False),
                },
            )

            # Use chunk_id as checksum for idempotency
            checksum = chunk_id
            if checksum in seen:
                skipped += 1
                continue

            seen.add(checksum)
            buffer.append((full_text, doc_id, checksum, chunk, CanonicalDocument(
                doc_id=doc_id,
                source_id=chunk_id,
                title=payload.get("title", ""),
                document_type="wikipedia",
                jurisdiction="VN",
                issued_date=None,
                effective_date=None,
                language="vi",
                content=text,
                source_url=payload.get("source_url", ""),
                checksum=checksum,
            )))
            bm25_docs.append((chunk_id, text))
            total_chunks += 1

            # Flush full batches
            while len(buffer) >= self.embed_batch_size:
                _submit_flush()

            # Log every 500 chunks
            if total_chunks % 500 == 0:
                self._log_embedding_progress(total_chunks, total_flushes, skipped, start_time)

        # Flush remaining buffer
        if buffer:
            _submit_flush()

        # Wait for all flushes
        _wait_for_flushes()
        executor.shutdown(wait=False)

        # Build BM25 index
        if bm25_docs:
            self.bm25_index.build(bm25_docs)

        elapsed = time.perf_counter() - start_time
        indexed = sum(1 for r in results if r.updated)
        print(f"\n{'='*60}")
        print(f" EMBED DONE — {total_chunks:,} chunks | {indexed:,} indexed | {skipped:,} skipped")
        print(f" {total_flushes} batches | BM25: {self.bm25_index.doc_count:,} docs")
        print(f" Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"{'='*60}")

        return results

    def _log_progress(self, total_docs: int, total_chunks: int, total_flushes: int, skipped: int, start_time: float) -> None:
        """Log progress to terminal."""
        if total_docs % 500 != 0:
            return

        elapsed = time.perf_counter() - start_time
        rate = total_docs / elapsed if elapsed > 0 else 0
        indexed = total_docs - skipped

        sys.stdout.write(f"\r\033[K")
        sys.stdout.write(
            f"[INGEST] {total_docs:>8,} docs | "
            f"{indexed:>7,} indexed | "
            f"{skipped:>6,} skip | "
            f"{total_chunks:>9,} chunks | "
            f"{total_flushes:>4} batches | "
            f"{elapsed:>6.0f}s | "
            f"{rate:>7.0f} docs/s"
        )
        sys.stdout.flush()

    def _log_chunking_progress(self, total_docs: int, total_chunks: int, skipped: int, start_time: float) -> None:
        """Log chunking progress to terminal."""
        elapsed = time.perf_counter() - start_time
        rate = total_docs / elapsed if elapsed > 0 else 0

        sys.stdout.write(f"\r\033[K")
        sys.stdout.write(
            f"[CHUNK] {total_docs:>8,} docs | "
            f"{total_chunks:>9,} chunks | "
            f"{skipped:>6,} skip | "
            f"{elapsed:>6.0f}s | "
            f"{rate:>7.0f} docs/s"
        )
        sys.stdout.flush()

    def _log_embedding_progress(self, total_chunks: int, total_flushes: int, skipped: int, start_time: float) -> None:
        """Log embedding progress to terminal."""
        elapsed = time.perf_counter() - start_time
        rate = total_chunks / elapsed if elapsed > 0 else 0

        sys.stdout.write(f"\r\033[K")
        sys.stdout.write(
            f"[EMBED] {total_chunks:>8,} chunks | "
            f"{total_flushes:>4} batches | "
            f"{skipped:>6,} skip | "
            f"{elapsed:>6.0f}s | "
            f"{rate:>7.0f} chunks/s"
        )
        sys.stdout.flush()

    @staticmethod
    def _raw_chunk_text(chunk: DocumentChunk) -> str:
        """Return raw chunk text without the contextual prefix if present.

        StructuredChunker prepends a context paragraph separated by a blank line.
        For keyword search we index only the original chunk content.
        """
        if "\n\n" in chunk.text:
            return chunk.text.split("\n\n", 1)[1]
        return chunk.text

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
