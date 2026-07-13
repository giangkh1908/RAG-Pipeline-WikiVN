"""Data models for the storage layer.

Hierarchy:
    Source → Document → Chunk → Index
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4


@dataclass
class Source:
    """An external data source."""

    tenant_id: str
    type: str
    version: str
    id: UUID = field(default_factory=uuid4)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A document belonging to a source."""

    source_id: UUID
    checksum: str
    status: str = "pending"  # pending, processing, indexed, failed
    id: UUID = field(default_factory=uuid4)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk belonging to a document."""

    document_id: UUID
    chunk_order: int
    content: str
    token_count: int
    id: UUID = field(default_factory=uuid4)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexEntry:
    """Vector index entry for a chunk.

    The actual vectors may be stored here (in-memory) or in an external
    vector database (Qdrant). This record keeps the mapping and metadata.
    """

    chunk_id: UUID
    dense_vector: list[float] | None = None
    sparse_vector: dict[int, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
