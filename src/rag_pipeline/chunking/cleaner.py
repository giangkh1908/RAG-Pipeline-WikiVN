"""Document cleaning stage."""

from __future__ import annotations

import re

from rag_pipeline.chunking.base import Cleaner
from rag_pipeline.chunking.models import CleanedDocument, NormalizedDocument


class DocumentCleaner(Cleaner):
    """Remove common markup and boilerplate from document content."""

    # Patterns for Wikipedia-like markup
    _REF_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
    _HTML_RE = re.compile(r"<[^>]+>", re.DOTALL)
    _WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")
    _BOLD_ITALIC_RE = re.compile(r"'''|''")
    _MAGIC_WORDS_RE = re.compile(r"__NOTOC__|__NOEDITSECTION__|__TOC__")

    # Innermost template first, repeated until stable
    _INNERMOST_TEMPLATE_RE = re.compile(r"\{\{[^{}]*?\}\}", re.DOTALL)

    def clean(self, document: NormalizedDocument) -> CleanedDocument:
        content = document.content

        content = self._remove_templates(content)
        content = self._remove_references_and_html(content)
        content = self._unwrap_wikilinks(content)
        content = self._strip_formatting(content)
        content = self._remove_magic_words(content)
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)

        return CleanedDocument(
            document_id=document.document_id,
            source_id=document.source_id,
            title=document.title,
            content=content.strip(),
            metadata=document.metadata,
        )

    def _remove_templates(self, text: str) -> str:
        for _ in range(30):
            new_text = self._INNERMOST_TEMPLATE_RE.sub(" ", text)
            if new_text == text:
                return new_text
            text = new_text
        return text

    def _remove_references_and_html(self, text: str) -> str:
        text = self._REF_RE.sub(" ", text)
        text = self._HTML_RE.sub(" ", text)
        return text

    def _unwrap_wikilinks(self, text: str) -> str:
        def _unwrap(match: re.Match) -> str:
            inner = match.group(1)
            parts = inner.split("|", 1)
            return parts[-1].strip()

        return self._WIKILINK_RE.sub(_unwrap, text)

    def _strip_formatting(self, text: str) -> str:
        return self._BOLD_ITALIC_RE.sub("", text)

    def _remove_magic_words(self, text: str) -> str:
        return self._MAGIC_WORDS_RE.sub(" ", text)
