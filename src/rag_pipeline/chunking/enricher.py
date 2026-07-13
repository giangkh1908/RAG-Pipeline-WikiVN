"""Metadata enrichment stage."""

from __future__ import annotations

from rag_pipeline.chunking.base import Enricher
from rag_pipeline.chunking.models import CleanedDocument, EnrichedDocument


class MetadataEnricher(Enricher):
    """Enrich document with inferred metadata and context hints.

    In a real system this may add:
    - inferred language
    - document type
    - source URL
    - parent/child relationships
    """

    def enrich(self, document: CleanedDocument) -> EnrichedDocument:
        metadata = dict(document.metadata)
        metadata.setdefault("title", document.title)
        metadata.setdefault("source_id", str(document.source_id))
        metadata.setdefault("document_id", str(document.document_id))

        return EnrichedDocument(
            document_id=document.document_id,
            source_id=document.source_id,
            title=document.title,
            content=document.content,
            metadata=metadata,
        )
