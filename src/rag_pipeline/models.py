from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class SourceRecord:
    source_id: str
    payload: dict[str, Any]


@dataclass(slots=True)
class QueryRecord:
    qid: str
    question: str
    context: list[str] = field(default_factory=list)
    cids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CanonicalDocument:
    doc_id: str
    source_id: str
    title: str
    document_type: str | None
    jurisdiction: str
    issued_date: str | None
    effective_date: str | None
    language: str
    content: str
    section_path: list[str] = field(default_factory=list)
    article_number: str | None = None
    clause_number: str | None = None
    version: str | None = None
    source_url: str | None = None
    checksum: str = ""
    ingest_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    section_path: list[str]
    article_number: str | None
    clause_number: str | None
    chunk_index: int
    token_count: int
    parent_chunk_id: str | None
    prev_chunk_id: str | None
    next_chunk_id: str | None
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IndexedChunk:
    chunk: DocumentChunk
    dense_vector: list[float]
    sparse_vector: dict[str, float] | None = None


@dataclass(slots=True)
class IndexingResult:
    document: CanonicalDocument
    chunks: list[DocumentChunk]
    updated: bool


@dataclass(slots=True)
class ProcessedQuery:
    """Output of Phase 2 query processing — ready for Phase 3 retrieval."""

    qid: str
    original_query: str
    normalized_query: str
    rewrite_query: str
    bm25_query: str
    intent: str = "general"  # definition, person, location, time, number, history, comparison, general
    filters: dict[str, Any] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Passage:
    """A single retrieved passage with scores."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_url: str = ""
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    rank: int = 0


@dataclass(slots=True)
class RetrievalResult:
    """Output of Phase 3 retrieval — ready for Phase 4 generation."""

    query: ProcessedQuery
    passages: list[Passage]
    context: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Citation:
    """A citation linking an answer claim to a source passage."""

    claim: str
    chunk_id: str
    doc_id: str
    title: str
    source_url: str = ""
    confidence: float = 0.0


@dataclass(slots=True)
class AnswerResult:
    """Final output of the RAG pipeline — Phase 4 generation."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    passages_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
