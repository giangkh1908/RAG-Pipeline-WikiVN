"""Models for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class RetrievalResult:
    """A single ranked result from hybrid retrieval."""

    chunk_id: UUID
    content: str
    rrf_score: float
    rank: int
    dense_score: float | None = None
    sparse_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
