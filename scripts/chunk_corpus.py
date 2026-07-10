"""Chunk the full UVW Wikipedia corpus into a JSONL file.

Usage:
    python scripts/chunk_corpus.py
    python scripts/chunk_corpus.py --sample 1000
    python scripts/chunk_corpus.py --output chunks/chunks_sample.jsonl --limit 1000

Output format (one JSON object per line):
    {
        "chunk_id": "...",
        "doc_id": "...",
        "title": "...",
        "source_url": "...",
        "section_path": ["Title", "Section", "Subsection"],
        "context": "This chunk is from the '...' document...",
        "text": "raw chunk content",
        "chunk_index": 0,
        "token_count": 123,
        "is_reference_section": false,
        "prev_chunk_id": null,
        "next_chunk_id": "..."
    }
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline.config import ChunkingConfig
from rag_pipeline.ingest.normalize import UVWWikipediaDocumentNormalizer
from rag_pipeline.indexing.bm25_index import BM25Index
from rag_pipeline.models import SourceRecord
from rag_pipeline.transform.cleaner import WikipediaArticleCleaner
from rag_pipeline.transform.structure_chunker import StructuredChunker


def iter_chunk_texts(chunk_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (chunk_id, raw_text) pairs from a chunk JSONL or JSONL.GZ file."""
    if chunk_path.suffix == ".gz":
        fin_ctx = gzip.open(chunk_path, "rt", encoding="utf-8")
    else:
        fin_ctx = open(chunk_path, "r", encoding="utf-8")

    with fin_ctx as fin:
        for line_no, line in enumerate(fin, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed chunk line {line_no}: {exc}", file=sys.stderr)
                continue

            chunk_id = payload.get("chunk_id")
            text = payload.get("text", "")
            if chunk_id and text:
                yield chunk_id, text


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk UVW Wikipedia corpus")
    parser.add_argument(
        "--input",
        default="documents/train.jsonl",
        help="Path to input JSONL corpus",
    )
    parser.add_argument(
        "--output",
        default="chunks/chunks.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N documents (for testing)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Process a random sample fraction (0.0-1.0), e.g. 0.5 for 50%.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle documents before processing (use with --sample)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Max tokens per chunk",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress every N documents",
    )
    parser.add_argument(
        "--bm25-output",
        default="index/bm25.pkl",
        help="Output path for BM25 index",
    )
    parser.add_argument(
        "--bm25-tokenizer",
        default="underthesea",
        choices=["underthesea", "pyvi", "simple"],
        help="Tokenizer for BM25 index",
    )
    parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="Skip BM25 index building",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunking_config = ChunkingConfig(max_tokens_per_chunk=args.max_tokens)
    normalizer = UVWWikipediaDocumentNormalizer(jurisdiction="VN", language="vi")
    cleaner = WikipediaArticleCleaner()
    chunker = StructuredChunker(chunking_config)

    total_docs = 0
    total_chunks = 0
    skipped_docs = 0
    start_time = time.perf_counter()

    # Pre-compute sampled line indices if requested
    line_indices: set[int] | None = None
    if args.sample is not None:
        import random

        if not (0.0 < args.sample <= 1.0):
            raise ValueError("--sample must be in (0.0, 1.0]")
        total_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
        sample_size = max(1, int(total_lines * args.sample))
        line_indices = set(random.sample(range(total_lines), sample_size))
        print(f"Sampling {sample_size:,} / {total_lines:,} documents ({args.sample*100:.2f}%)")

    # Choose plain or gzip output
    if output_path.suffix == ".gz":
        fout_ctx = gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6)
    else:
        fout_ctx = open(output_path, "w", encoding="utf-8")

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        fout_ctx as fout,
    ):
        for line_no, line in enumerate(fin, start=1):
            if args.limit is not None and total_docs >= args.limit:
                break
            if line_indices is not None and (line_no - 1) not in line_indices:
                continue

            total_docs += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed line {line_no}: {exc}", file=sys.stderr)
                skipped_docs += 1
                continue

            source_id = payload.get("id", f"line_{line_no}")
            record = SourceRecord(source_id=source_id, payload=payload)
            document = normalizer.normalize(record)

            # Skip Wikipedia module / template pages (Lua code, not prose)
            if document.title.startswith(("Mô đun:", "Module:")):
                skipped_docs += 1
                continue

            document.content = cleaner.clean(document.content)
            chunks = chunker.chunk(document)

            if not chunks:
                skipped_docs += 1

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
            if total_docs % args.log_every == 0:
                elapsed = time.perf_counter() - start_time
                docs_per_sec = total_docs / elapsed if elapsed > 0 else 0
                print(
                    f"[PROGRESS] {total_docs:,} docs | {total_chunks:,} chunks | "
                    f"{skipped_docs:,} skipped | {docs_per_sec:.1f} docs/s | "
                    f"elapsed {elapsed:.1f}s",
                    flush=True,
                )

    elapsed = time.perf_counter() - start_time
    print("\n=== Chunking complete ===", flush=True)
    print(f"Input documents: {total_docs:,}", flush=True)
    print(f"Output chunks:   {total_chunks:,}", flush=True)
    print(f"Skipped docs:    {skipped_docs:,}", flush=True)
    print(f"Elapsed time:    {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Output file:     {output_path.resolve()}", flush=True)

    # Build BM25 index from collected chunks
    if not args.no_bm25 and total_chunks > 0:
        bm25_path = Path(args.bm25_output)
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nBuilding BM25 index ({total_chunks:,} chunks)...", flush=True)
        bm25_start = time.perf_counter()
        bm25_index = BM25Index(index_path=bm25_path, tokenizer_name=args.bm25_tokenizer)
        bm25_index.build(iter_chunk_texts(output_path))
        bm25_elapsed = time.perf_counter() - bm25_start
        print(f"BM25 built: {bm25_index.doc_count:,} docs indexed", flush=True)
        print(f"BM25 file:  {bm25_path.resolve()}", flush=True)
        print(f"BM25 time:  {bm25_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
