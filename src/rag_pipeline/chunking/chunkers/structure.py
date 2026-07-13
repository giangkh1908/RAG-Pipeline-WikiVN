"""Structure-aware chunker.

Groups blocks within detected sections, keeping lists together and adding
section context to each chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rag_pipeline.chunking.base import Chunker
from rag_pipeline.chunking.models import ChunkCandidate, SectionedDocument


@dataclass
class StructureChunker(Chunker):
    """Chunk by structure: paragraphs and list blocks, respecting section boundaries."""

    max_tokens: int = 300
    add_context_prefix: bool = True

    _WORD_RE = re.compile(r"\S+")
    _REFERENCE_HEADINGS: frozenset[str] = frozenset(
        {
            "tham khảo",
            "liên kết ngoài",
            "xem thêm",
            "chú thích",
            "tài liệu tham khảo",
            "nguồn tham khảo",
            "đọc thêm",
            "tham khảo thêm",
            "ghi chú",
            "chú giải",
            "trích dẫn",
            "references",
            "see also",
            "external links",
            "notes",
            "footnotes",
            "further reading",
            "bibliography",
        }
    )

    def chunk(self, document: SectionedDocument) -> list[ChunkCandidate]:
        candidates: list[ChunkCandidate] = []
        global_order = 0

        for section in document.sections:
            section_path = [document.title, section.title] if section.title else [document.title]
            is_reference = self._is_reference_heading(section.title)
            blocks = self._parse_blocks(section.content)
            groups = self._group_blocks(blocks)

            for group in groups:
                content = self._build_chunk_text(document.title, section_path, group, is_reference)
                candidates.append(
                    ChunkCandidate(
                        document_id=document.document_id,
                        chunk_order=global_order,
                        content=content,
                        token_count=self._count_tokens(content),
                        section_path=section_path,
                        metadata={
                            "section_title": section.title,
                            "is_reference_section": is_reference,
                        },
                    )
                )
                global_order += 1

        return candidates

    def _parse_blocks(self, text: str) -> list[tuple[Literal["paragraph", "list"], str]]:
        """Parse section content into paragraph and list blocks."""
        blocks: list[tuple[Literal["paragraph", "list"], str]] = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
            if not lines:
                continue

            if all(self._is_list_line(ln) for ln in lines):
                blocks.append(("list", para))
            else:
                blocks.append(("paragraph", para))

        return blocks

    def _group_blocks(
        self, blocks: list[tuple[Literal["paragraph", "list"], str]]
    ) -> list[list[tuple[Literal["paragraph", "list"], str]]]:
        """Group consecutive blocks into chunks respecting max_tokens."""
        if not blocks:
            return []

        groups: list[list[tuple[Literal["paragraph", "list"], str]]] = []
        current: list[tuple[Literal["paragraph", "list"], str]] = [blocks[0]]
        current_tokens = self._count_tokens(blocks[0][1])

        for block in blocks[1:]:
            block_tokens = self._count_tokens(block[1])
            # Keep list blocks in their own group if current is not empty
            if block[0] == "list" and current:
                groups.append(current)
                current = [block]
                current_tokens = block_tokens
                continue

            if current_tokens + block_tokens <= self.max_tokens:
                current.append(block)
                current_tokens += block_tokens
            else:
                groups.append(current)
                current = [block]
                current_tokens = block_tokens

        if current:
            groups.append(current)

        return groups

    def _build_chunk_text(
        self,
        title: str,
        section_path: list[str],
        group: list[tuple[Literal["paragraph", "list"], str]],
        is_reference: bool,
    ) -> str:
        raw_text = "\n\n".join(block[1] for block in group)

        if not self.add_context_prefix:
            return raw_text

        context = self._build_context(title, section_path)
        return f"{context}\n\n{raw_text}"

    def _build_context(self, title: str, section_path: list[str]) -> str:
        ctx = f"This chunk is from the '{title}' document"
        if len(section_path) > 1:
            sec = " > ".join(section_path[1:])
            ctx += f", specifically the '{sec}' section"
        return ctx + "."

    def _is_list_line(self, line: str) -> bool:
        return bool(re.match(r"^[*#\-]|\d+[.)]\s", line.strip()))

    def _is_reference_heading(self, title: str) -> bool:
        return title.strip().lower() in self._REFERENCE_HEADINGS

    def _count_tokens(self, text: str) -> int:
        return len(self._WORD_RE.findall(text))
