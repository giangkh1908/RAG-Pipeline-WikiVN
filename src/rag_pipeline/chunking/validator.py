"""Chunk validation stage."""

from __future__ import annotations

from rag_pipeline.chunking.base import Validator
from rag_pipeline.chunking.models import ChunkCandidate


class ChunkValidator(Validator):
    """Validate chunk candidates and drop invalid ones."""

    def __init__(self, min_tokens: int = 20, max_tokens: int = 2000) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def validate(self, candidates: list[ChunkCandidate]) -> list[ChunkCandidate]:
        valid: list[ChunkCandidate] = []
        for candidate in candidates:
            if self._is_valid(candidate):
                valid.append(candidate)
        return valid

    def _is_valid(self, candidate: ChunkCandidate) -> bool:
        if not candidate.content or not candidate.content.strip():
            return False
        if candidate.token_count < self.min_tokens:
            return False
        if candidate.token_count > self.max_tokens:
            return False
        return True
