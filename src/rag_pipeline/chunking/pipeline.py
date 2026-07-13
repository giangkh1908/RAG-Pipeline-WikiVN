"""Chunking pipeline orchestrator."""

from __future__ import annotations

from rag_pipeline.chunking.base import Chunker
from rag_pipeline.chunking.cleaner import DocumentCleaner
from rag_pipeline.chunking.enricher import MetadataEnricher
from rag_pipeline.chunking.models import RawDocument
from rag_pipeline.chunking.normalizer import DocumentNormalizer
from rag_pipeline.chunking.section_detector import HeadingSectionDetector
from rag_pipeline.chunking.validator import ChunkValidator
from rag_pipeline.storage.models import Chunk


class ChunkingPipeline:
    """Raw → Normalize → Clean → Enrich → Section Detect → Chunk → Validate."""

    def __init__(
        self,
        chunker: Chunker,
        normalizer: DocumentNormalizer | None = None,
        cleaner: DocumentCleaner | None = None,
        enricher: MetadataEnricher | None = None,
        section_detector: HeadingSectionDetector | None = None,
        validator: ChunkValidator | None = None,
    ) -> None:
        self.normalizer = normalizer or DocumentNormalizer()
        self.cleaner = cleaner or DocumentCleaner()
        self.enricher = enricher or MetadataEnricher()
        self.section_detector = section_detector or HeadingSectionDetector()
        self.chunker = chunker
        self.validator = validator or ChunkValidator()

    def process(self, document: RawDocument) -> list[Chunk]:
        """Run the full chunking pipeline and return validated storage chunks."""
        normalized = self.normalizer.normalize(document)
        cleaned = self.cleaner.clean(normalized)
        enriched = self.enricher.enrich(cleaned)
        sectioned = self.section_detector.detect(enriched)
        candidates = self.chunker.chunk(sectioned)
        validated = self.validator.validate(candidates)

        return [
            Chunk(
                document_id=c.document_id,
                chunk_order=c.chunk_order,
                content=c.content,
                token_count=c.token_count,
                metadata={
                    "section_path": c.section_path,
                    **c.metadata,
                },
            )
            for c in validated
        ]
