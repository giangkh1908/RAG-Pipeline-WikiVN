from __future__ import annotations

from datetime import datetime
from typing import Any

from rag_pipeline.models import CanonicalDocument, SourceRecord
from rag_pipeline.utils.hashing import stable_hash


class UVWWikipediaDocumentNormalizer:
    """Maps UVW-2026 article rows into the canonical document contract."""

    TITLE_KEYS = ("title", "name")
    CONTENT_KEYS = ("content", "text", "body")

    def __init__(self, jurisdiction: str = "VN", language: str = "vi") -> None:
        self.jurisdiction = jurisdiction
        self.language = language

    def normalize(self, record: SourceRecord) -> CanonicalDocument:
        payload = record.payload
        article_id = self._string_or_none(payload.get("id")) or record.source_id
        title = self._first_value(payload, self.TITLE_KEYS) or article_id
        content = self._first_value(payload, self.CONTENT_KEYS) or ""
        wikidata_id = self._string_or_none(payload.get("wikidata_id"))
        main_category = self._string_or_none(payload.get("main_category"))
        quality_score = self._int_or_none(payload.get("quality"))
        if quality_score is None:
            quality_score = self._int_or_none(payload.get("quality_score"))
        num_chars = self._int_or_none(payload.get("num_chars"))
        num_sentences = self._int_or_none(payload.get("num_sentences"))
        source_url = f"https://vi.wikipedia.org/wiki/{article_id}"

        checksum = stable_hash(
            {
                "source_id": article_id,
                "title": title,
                "content": content,
                "document_type": "wikipedia_article",
                "source_url": source_url,
                "wikidata_id": wikidata_id,
                "main_category": main_category,
                "quality_score": quality_score,
            }
        )

        return CanonicalDocument(
            doc_id=checksum[:24],
            source_id=article_id,
            title=title,
            document_type="wikipedia_article",
            jurisdiction=self.jurisdiction,
            issued_date=None,
            effective_date=None,
            language=self.language,
            content=content,
            section_path=[title] if title else [],
            article_number=None,
            clause_number=None,
            version=wikidata_id,
            source_url=source_url,
            checksum=checksum,
            ingest_timestamp=datetime.utcnow(),
            metadata={
                "dataset": "undertheseanlp/UVW-2026",
                "article_id": article_id,
                "wikidata_id": wikidata_id,
                "main_category": main_category,
                "quality_score": quality_score,
                "num_chars": num_chars,
                "num_sentences": num_sentences,
            },
        )

    def _first_value(self, payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def _string_or_none(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _int_or_none(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
