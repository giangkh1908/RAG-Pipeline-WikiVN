from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ChunkingConfig:
    max_tokens_per_chunk: int = 300
    chunk_overlap_tokens: int = 40
    min_chunk_tokens: int = 40
    chunking_strategy: str = "recursive"


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_seconds: float = 30.0
    sub_batch_size: int = 500
    max_retries: int = 3
    parallel_workers: int = 4


@dataclass(slots=True)
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection_name: str = "wikipedia_vi_chunks"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "bm25"


@dataclass(slots=True)
class LLMConfig:
    model_name: str = "deepseek/deepseek-v4-flash"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    temperature: float = 0.1
    max_tokens: int = 512


@dataclass(slots=True)
class QueryConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    enable_rewrite: bool = True
    enable_guardrails: bool = True
    max_query_length: int = 500


@dataclass(slots=True)
class RetrievalConfig:
    # Dense search
    dense_top_k: int = 50
    # BM25 search
    bm25_top_k: int = 50
    bm25_index_path: Path = Path("index/bm25.pkl")
    bm25_tokenizer: str = "underthesea"  # "underthesea", "pyvi", or "simple"
    # RRF fusion
    rrf_k: int = 60
    rrf_top_k: int = 20
    # Re-ranking
    enable_rerank: bool = True
    rerank_provider: str = "cohere"  # "cohere" or "bge"
    rerank_model: str = "rerank-v3.5"
    rerank_api_key_env: str = "COHERE_API_KEY"
    rerank_top_k: int = 5
    # Score thresholds
    min_score: float = 0.0


@dataclass(slots=True)
class IngestConfig:
    source_type: str = "local_jsonl"
    jsonl_path: Path = Path("documents/train.jsonl")
    jsonl_sample_percent: float = 100.0
    hf_dataset_name: str = "undertheseanlp/UVW-2026"
    hf_dataset_split: str = "train"
    hf_sample_percent: float = 1.0
    hf_min_quality_score: int | None = None
    corpus_path: Path = Path("document/corpus.csv")
    train_queries_path: Path = Path("document/train.csv")
    train_split_queries_path: Path = Path("document/train_split.csv")
    validation_queries_path: Path = Path("document/val_split.csv")
    public_test_queries_path: Path = Path("document/public_test.csv")
    language: str = "vi"
    jurisdiction: str = "VN"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
