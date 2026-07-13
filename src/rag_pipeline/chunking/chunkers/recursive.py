"""Recursive chunker: paragraph → sentence → word."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_pipeline.chunking.base import Chunker
from rag_pipeline.chunking.models import ChunkCandidate, Section, SectionedDocument


@dataclass
class RecursiveChunker(Chunker):
    """Split sections by paragraph, then sentence, then word boundary."""

    max_tokens: int = 300
    chunk_overlap: int = 40

    _SENTENCE_RE = re.compile(r"(?<=[.!?;…])\s+")
    _WORD_RE = re.compile(r"\S+")

    def chunk(self, document: SectionedDocument) -> list[ChunkCandidate]:
        candidates: list[ChunkCandidate] = []
        global_order = 0

        for section in document.sections:
            section_path = [document.title, section.title] if section.title else [document.title]
            pieces = self._split_section(section)

            for piece in pieces:
                candidates.append(
                    ChunkCandidate(
                        document_id=document.document_id,
                        chunk_order=global_order,
                        content=piece,
                        token_count=self._count_tokens(piece),
                        section_path=section_path,
                        metadata={"section_title": section.title},
                    )
                )
                global_order += 1

        return candidates

    def _split_section(self, section: Section) -> list[str]:
        paragraphs = [p.strip() for p in section.content.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        merged = self._merge_pieces(paragraphs)
        result: list[str] = []
        for piece in merged:
            result.extend(self._split_to_fit(piece))
        return result

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        if not pieces:
            return []

        merged: list[str] = []
        current = pieces[0]
        current_tokens = self._count_tokens(current)

        for piece in pieces[1:]:
            piece_tokens = self._count_tokens(piece)
            if current_tokens + piece_tokens <= self.max_tokens:
                current = f"{current}\n\n{piece}"
                current_tokens += piece_tokens
            else:
                merged.append(current)
                current = piece
                current_tokens = piece_tokens

        merged.append(current)
        return merged

    def _split_to_fit(self, text: str) -> list[str]:
        tokens = self._count_tokens(text)
        if tokens <= self.max_tokens:
            return [text]

        sentences = self._SENTENCE_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 1:
            merged = self._merge_pieces(sentences)
            result: list[str] = []
            for piece in merged:
                result.extend(self._split_by_words(piece))
            return result

        return self._split_by_words(text)

    def _split_by_words(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.max_tokens:
            return [text]

        step = max(1, self.max_tokens - self.chunk_overlap)
        pieces: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.max_tokens, len(words))
            pieces.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += step
        return pieces

    def _count_tokens(self, text: str) -> int:
        return len(self._WORD_RE.findall(text))
