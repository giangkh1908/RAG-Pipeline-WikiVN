# Retrieval Pipeline (Phase 3) — v1

## Tổng quan

> **v1**: Hybrid search (dense + BM25) → RRF fusion → Cohere re-rank. Single query, không có multi-step.

Hybrid search (dense + BM25) → RRF fusion → Cohere re-rank → top-k results.

## Pipeline flow

```
ProcessedQuery (từ Phase 2)
    ├─ Dense search: embed rewrite_query → Qdrant top_k=50
    ├─ BM25 search: bm25_query → BM25 index top_k=50
    └─ RRF fusion: merge 2 kết quả → top_k=20
    → Cohere Re-ranker: re-rank top 20 → top 5
    → RetrievalResult (passages + context)
```

## Cấu trúc thư mục

```
src/rag_pipeline/
├── config.py                    # RetrievalConfig
├── models.py                    # Passage, RetrievalResult
├── indexing/
│   ├── bm25_index.py            # BM25 build/search/save/load
│   ├── rrf.py                   # Reciprocal Rank Fusion
│   ├── reranker.py              # CohereReranker, BGEReranker, TestReranker
│   └── vector_store.py          # + search() method
└── pipelines/
    └── retrieval_pipeline.py    # RetrievalPipeline orchestrator
```

## Components

### 1. Dense Vector Search (`indexing/vector_store.py`)

- Embed `rewrite_query` bằng OpenRouter embedding (2048-dim)
- Search Qdrant: cosine similarity, top_k=50
- Implement `search()` trong `QdrantVectorStore` và `InMemoryVectorStore`

### 2. BM25 Index (`indexing/bm25_index.py`)

- Backend: SQLite FTS5 (instant load, incremental, no RAM)
- Tokenizer: underthesea (Vietnamese word segmentation), pyvi, hoặc simple (fallback)
- Index: raw content only (không chứa context prefix)
- Incremental insert: INSERT vào SQLite khi ingest
- Search: trả về (chunk_id, doc_id, score, full_text)

### 3. RRF Fusion (`indexing/rrf.py`)

Reciprocal Rank Fusion:
```
score(d) = Σ 1/(k + rank_i(d))
```
- k=60 (standard value)
- Merge dense + BM25 results
- Sort theo RRF score descending

### 4. Cohere Re-ranker (`indexing/reranker.py`)

API: `https://api.cohere.com/v2/rerank`

**Config:**
```python
@dataclass
class CohereReranker:
    model_name: str = "rerank-v3.5"        # multilingual, hỗ trợ tiếng Việt
    api_base: str = "https://api.cohere.com/v2/rerank"
    api_key_env: str = "COHERE_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 3
```

**Free tier:** 100 search units/tháng

**Request:**
```json
{
    "model": "rerank-v3.5",
    "query": "Thủ đô Việt Nam ở đâu?",
    "documents": ["passage 1", "passage 2", ...],
    "top_n": 5
}
```

**Response:**
```json
{
    "results": [
        {"index": 3, "relevance_score": 0.999},
        {"index": 0, "relevance_score": 0.850}
    ]
}
```

### 5. RetrievalPipeline (`pipelines/retrieval_pipeline.py`)

Orchestrator:
```python
@dataclass
class RetrievalPipeline:
    config: RetrievalConfig
    embedder: Embedder
    vector_store: VectorStore
    bm25_index: BM25Index
    reranker: CohereReranker | None = None

    def run(self, query: ProcessedQuery) -> RetrievalResult
```

## Config

```python
@dataclass
class RetrievalConfig:
    # Dense search
    dense_top_k: int = 50
    # BM25 search
    bm25_top_k: int = 50
    bm25_index_path: Path = Path("index/bm25.db")
    bm25_tokenizer: str = "underthesea"
    # RRF fusion
    rrf_k: int = 60
    rrf_top_k: int = 20
    # Cohere re-ranking
    enable_rerank: bool = True
    rerank_provider: str = "cohere"      # "cohere" or "bge"
    rerank_model: str = "rerank-v3.5"
    rerank_api_key_env: str = "COHERE_API_KEY"
    # Score thresholds
    min_score: float = 0.0
```

## Output: RetrievalResult

```python
@dataclass
class Passage:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_url: str
    dense_score: float      # cosine similarity
    bm25_score: float       # BM25 score
    rrf_score: float        # RRF fusion score
    rerank_score: float     # Cohere relevance score
    rank: int               # final rank (1-5)

@dataclass
class RetrievalResult:
    query: ProcessedQuery
    passages: list[Passage]     # top 5 passages
    context: str                # assembled context string
    metadata: dict[str, Any]    # dense_count, bm25_count, fused_count
```

## CLI

```powershell
# Search with Cohere re-rank
python -m rag_pipeline.main search --question "Thủ đô Việt Nam ở đâu?" --rerank

# Search without re-rank
python -m rag_pipeline.main search --question "..."

# Search with InMemory (test)
python -m rag_pipeline.main search --question "..." --no-qdrant
```

## Dependencies

```bash
pip install underthesea      # Vietnamese tokenizer (optional, for BM25)
pip install httpx            # HTTP client (already installed)
# SQLite FTS5 — built-in Python, no extra install
```

## Environment Variables

```env
OPENROUTER_API_KEY=sk-or-...    # Embedding + LLM
COHERE_API_KEY=...              # Re-ranker
```

## Tests

| File | Tests | Coverage |
|------|-------|----------|
| test_bm25_index.py | 10 | Insert, search, save/load, tokenizer, incremental |
| test_rrf.py | 7 | Fusion logic, score calculation, BM25-only results |
| test_retrieval_pipeline.py | 4 | Full pipeline, empty store, context assembly |
| **Total** | **21** | **132/132 pass** (bao gồm Phase 1+2 tests) |
