"""Recursive paragraph-based chunker.

Splits text by paragraph → sentence → word boundary.
No heavy ML dependencies — fast enough for 1M+ documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_pipeline.config import ChunkingConfig
from rag_pipeline.models import CanonicalDocument, DocumentChunk
from rag_pipeline.utils.hashing import stable_hash

# Vietnamese sentence-ending punctuation
_SENTENCE_RE = re.compile(r"(?<=[.!?;…])\s+")
_WORD_RE = re.compile(r"\S+")


@dataclass
class RecursiveChunker:
    """Split documents by paragraph → sentence → word, respecting max token limit."""

    config: ChunkingConfig

    def chunk(self, document: CanonicalDocument) -> list[DocumentChunk]:
        raw_chunks = self._recursive_split(document.content)
        if not raw_chunks:
            return []

        chunks: list[DocumentChunk] = []
        for idx, text in enumerate(raw_chunks):
            chunks.append(self._make_chunk(document, text, idx))

        self._link_neighbors(chunks)
        return chunks

    # ── Core splitting logic ───────────────────────────────────────

    def _recursive_split(self, text: str) -> list[str]:
        """Top-level: split by paragraph, merge small ones, then recurse into oversized pieces."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return []

        # Merge consecutive small paragraphs first
        merged = self._merge_pieces(paragraphs, self._count_tokens)
        result: list[str] = []
        for piece in merged:
            result.extend(self._split_node(piece, depth=0))
        return result

    def _split_node(self, text: str, depth: int) -> list[str]:
        """Recursively split text until each piece fits within max_tokens."""
        token_count = self._count_tokens(text)
        if token_count <= self.config.max_tokens_per_chunk:
            return [text]

        # Depth 0: try splitting by sentence
        # Depth 1: split by word (sliding window)
        if depth == 0:
            return self._split_by_sentence(text)
        return self._split_by_words(text)

    def _split_by_sentence(self, text: str) -> list[str]:
        """Split by sentence boundaries, merge small sentences into chunks."""
        sentences = _SENTENCE_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text]

        return self._merge_pieces(sentences, self._count_tokens)

    def _split_by_words(self, text: str) -> list[str]:
        """Sliding window split by word count (last resort)."""
        words = text.split()
        if len(words) <= self.config.max_tokens_per_chunk:
            return [text]

        step = max(1, self.config.max_tokens_per_chunk - self.config.chunk_overlap_tokens)
        pieces: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.config.max_tokens_per_chunk, len(words))
            piece = " ".join(words[start:end]).strip()
            if piece:
                pieces.append(piece)
            if end == len(words):
                break
            start += step
        return pieces

    def _merge_pieces(self, pieces: list[str], count_fn) -> list[str]:
        """Greedy merge small pieces until hitting max_tokens."""
        chunks: list[str] = []
        current = pieces[0]
        current_count = count_fn(current)

        for piece in pieces[1:]:
            piece_count = count_fn(piece)
            if current_count + piece_count <= self.config.max_tokens_per_chunk:
                current = f"{current} {piece}"
                current_count += piece_count
            else:
                # If current alone is too big, recursively split it
                if current_count > self.config.max_tokens_per_chunk:
                    chunks.extend(self._split_node(current, depth=1))
                else:
                    chunks.append(current)
                current = piece
                current_count = piece_count

        if current:
            if current_count > self.config.max_tokens_per_chunk:
                chunks.extend(self._split_node(current, depth=1))
            else:
                chunks.append(current)

        return chunks

    # ── Helpers ────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """Estimate token count by word splitting (fast, good enough for Vietnamese)."""
        return len(_WORD_RE.findall(text))

    def _make_chunk(self, document: CanonicalDocument, text: str, index: int) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=stable_hash({"doc_id": document.doc_id, "chunk_index": index, "text": text})[:24],
            doc_id=document.doc_id,
            text=text,
            section_path=document.section_path or [document.title],
            article_number=document.article_number,
            clause_number=document.clause_number,
            chunk_index=index,
            token_count=self._count_tokens(text),
            parent_chunk_id=None,
            prev_chunk_id=None,
            next_chunk_id=None,
            checksum=stable_hash({"doc_id": document.doc_id, "text": text}),
            metadata={"title": document.title, "source_url": document.source_url, **document.metadata},
        )

    def _link_neighbors(self, chunks: list[DocumentChunk]) -> None:
        for i, c in enumerate(chunks):
            c.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
            c.next_chunk_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None
