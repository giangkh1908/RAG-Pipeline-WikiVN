"""Section detection stage."""

from __future__ import annotations

import re

from rag_pipeline.chunking.base import SectionDetector
from rag_pipeline.chunking.models import EnrichedDocument, Section, SectionedDocument


class HeadingSectionDetector(SectionDetector):
    """Detect sections from Markdown-style headings or standalone heading lines."""

    _MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+)$", re.MULTILINE)
    _WORD_RE = re.compile(r"\S+")

    def detect(self, document: EnrichedDocument) -> SectionedDocument:
        if not document.content:
            return SectionedDocument(
                document_id=document.document_id,
                source_id=document.source_id,
                title=document.title,
                sections=[],
                metadata=document.metadata,
            )

        # Try markdown headings first
        if self._has_markdown_headings(document.content):
            sections = self._parse_markdown_headings(document.content)
        else:
            sections = self._parse_implicit_headings(document.content)

        if not sections:
            sections = [Section(title=document.title, level=1, content=document.content)]

        return SectionedDocument(
            document_id=document.document_id,
            source_id=document.source_id,
            title=document.title,
            sections=sections,
            metadata=document.metadata,
        )

    def _has_markdown_headings(self, text: str) -> bool:
        return bool(self._MARKDOWN_HEADING_RE.search(text))

    def _parse_markdown_headings(self, text: str) -> list[Section]:
        sections: list[Section] = []
        current_title = ""
        current_level = 1
        current_lines: list[str] = []

        for line in text.split("\n"):
            match = self._MARKDOWN_HEADING_RE.match(line)
            if match:
                if current_lines:
                    sections.append(self._make_section(current_title, current_level, current_lines))
                current_title = match.group(2).strip()
                current_level = len(match.group(1))
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(self._make_section(current_title, current_level, current_lines))

        return sections

    def _parse_implicit_headings(self, text: str) -> list[Section]:
        """Fallback: treat short standalone uppercase lines as headings."""
        sections: list[Section] = []
        current_title = ""
        current_lines: list[str] = []

        for line in text.split("\n"):
            stripped = line.strip()
            if self._looks_like_heading(stripped):
                if current_lines:
                    sections.append(self._make_section(current_title, 1, current_lines))
                current_title = stripped
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(self._make_section(current_title, 1, current_lines))

        return sections

    def _looks_like_heading(self, line: str) -> bool:
        if not line:
            return False
        words = self._WORD_RE.findall(line)
        if len(words) > 10 or len(line) > 100:
            return False
        if re.search(r"[.!?;:,]$", line):
            return False
        if not re.match(r"^[A-ZÀ-ỸĐ0-9]", line):
            return False
        return True

    def _make_section(self, title: str, level: int, lines: list[str]) -> Section:
        content = "\n".join(lines).strip()
        return Section(title=title or "", level=level, content=content)
