from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(slots=True)
class ChunkingConfig:
    """Configuration for document chunking."""

    max_tokens: int = 300
    chunk_overlap: int = 40
    min_tokens: int = 20


@dataclass(slots=True)
class DenseEmbeddingConfig:
    """Configuration for dense embedding via OpenRouter."""

    model_name: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    batch_size: int = 32


@dataclass(slots=True)
class SparseEmbeddingConfig:
    """Configuration for classic BM25 sparse embedding."""

    model_name: str = "classic_bm25"
    k: float = 1.5
    b: float = 0.75
    avg_len: float = 256.0
    vocab_path: str = "data/bm25_vocab.json"
    batch_size: int = 256


@dataclass(slots=True)
class QdrantConfig:
    """Configuration for Qdrant vector database."""

    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection_name: str = "rag_chunks_v2"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"
    dense_top_k: int = 25
    sparse_top_k: int = 25
    on_disk: bool = True
    # Set to "idf" when using raw term frequencies (e.g. FastEmbed BM25).
    # Set to None when vectors already contain BM25 weights.
    sparse_modifier: str | None = None


@dataclass(slots=True)
class LLMQueryConfig:
    """Configuration for LLM-based query preprocessing."""

    model_name: str = "deepseek/deepseek-v4-flash"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    prompt_version: str = "v1"
    cache_ttl_days: int = 30
    fallback_to_normalized: bool = True


@dataclass(slots=True)
class ContextBuilderConfig:
    """Configuration for building context from retrieval results."""

    max_chunks: int = 5
    include_title: bool = True
    citation_format: str = "[{id}]"


@dataclass(slots=True)
class GenerationConfig:
    """Configuration for LLM answer generation."""

    model_name: str = "openai/gpt-4o-mini"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    max_tokens: int = 800
    temperature: float = 0.3
    timeout_seconds: float = 60.0
    max_retries: int = 3


@dataclass(slots=True)
class MemoryConfig:
    """Configuration for chat memory (session-based, no auth).

    The compact threshold is computed at runtime from
    ``keep_raw_turns * (max_input_tokens + max_output_tokens) * 0.7``
    so that adjusting input/output limits rescales the threshold
    automatically.
    """

    enabled: bool = True
    keep_raw_turns: int = 3
    summary_max_tokens: int = 256
    summary_temperature: float = 0.2
    summary_model_name: str = "openai/gpt-4o-mini"
    summary_api_base: str = "https://openrouter.ai/api/v1"
    summary_api_key_env: str = "OPENROUTER_API_KEY"
    summary_max_retries: int = 3
    summary_timeout_seconds: float = 60.0
    session_ttl_hours: int = 24
    max_input_chars: int = 500
    max_output_tokens: int = 800
    char_per_token: int = 3  # heuristic divisor for Vietnamese


@dataclass(slots=True)
class StorageConfig:
    """Configuration for the relational storage backend."""

    db_path: str = "data/rag_storage.db"


@dataclass(slots=True)
class RetrievalConfig:
    """Top-level configuration for retrieval."""

    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    dense: DenseEmbeddingConfig = field(default_factory=DenseEmbeddingConfig)
    sparse: SparseEmbeddingConfig = field(default_factory=SparseEmbeddingConfig)
    llm_query: LLMQueryConfig = field(default_factory=LLMQueryConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    rrf_k: int = 60
    rrf_top_k: int = 20


@dataclass(slots=True)
class RAGConfig:
    """Top-level configuration for the full RAG pipeline."""

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    context_builder: ContextBuilderConfig = field(default_factory=ContextBuilderConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
