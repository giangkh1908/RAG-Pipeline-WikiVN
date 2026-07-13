"""Protocols for the chunking pipeline."""

from __future__ import annotations

from typing import Protocol

from rag_pipeline.chunking.models import (
    ChunkCandidate,
    CleanedDocument,
    EnrichedDocument,
    NormalizedDocument,
    RawDocument,
    SectionedDocument,
)


class Normalizer(Protocol):
    def normalize(self, document: RawDocument) -> NormalizedDocument: ...


class Cleaner(Protocol):
    def clean(self, document: NormalizedDocument) -> CleanedDocument: ...


class Enricher(Protocol):
    def enrich(self, document: CleanedDocument) -> EnrichedDocument: ...


class SectionDetector(Protocol):
    def detect(self, document: EnrichedDocument) -> SectionedDocument: ...


class Chunker(Protocol):
    """Pluggable chunker. Operates on a sectioned document."""

    def chunk(self, document: SectionedDocument) -> list[ChunkCandidate]: ...


class Validator(Protocol):
    def validate(self, candidates: list[ChunkCandidate]) -> list[ChunkCandidate]: ...
