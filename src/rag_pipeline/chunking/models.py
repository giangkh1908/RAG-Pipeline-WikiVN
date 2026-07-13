"""Models for the chunking pipeline.

Pipeline stages:
    RawDocument → NormalizedDocument → CleanedDocument → EnrichedDocument
    → SectionedDocument → ChunkCandidate → storage.models.Chunk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class RawDocument:
    """Input to the chunking pipeline."""

    document_id: UUID
    source_id: UUID
    title: str
    raw_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedDocument:
    """After unicode normalization and line-ending repair."""

    document_id: UUID
    source_id: UUID
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanedDocument:
    """After removing markup, references, and boilerplate."""

    document_id: UUID
    source_id: UUID
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedDocument:
    """After adding inferred metadata and context hints."""

    document_id: UUID
    source_id: UUID
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    """A detected section within a document."""

    title: str
    level: int
    content: str


@dataclass
class SectionedDocument:
    """After section detection."""

    document_id: UUID
    source_id: UUID
    title: str
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkCandidate:
    """Candidate chunk before validation."""

    document_id: UUID
    chunk_order: int
    content: str
    token_count: int
    section_path: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
