"""Models for indexing and search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class SearchResult:
    """A single result from vector search."""

    chunk_id: UUID
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
