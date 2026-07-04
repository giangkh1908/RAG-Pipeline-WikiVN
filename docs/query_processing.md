# Query Processing Pipeline (Phase 2) — v1

## Tổng quan

> **v1**: Normalize → Guardrails → LLM rewrite (với conversation context cho pronoun resolution).

Xử lý user query trước khi retrieval: normalize → guardrails → rewrite → output `ProcessedQuery`.

## Cấu trúc thư mục

```
src/rag_pipeline/
├── config.py                    # QueryConfig, LLMConfig
├── models.py                    # ProcessedQuery
├── query/
│   ├── __init__.py
│   ├── normalizer.py            # Vietnamese text normalization
│   ├── rewriter.py              # LLM-based query rewrite
│   └── guardrails.py            # Input safety checks
├── indexing/
│   ├── llm_client.py            # OpenRouter LLM client
│   └── vector_store.py          # + search() method
└── pipelines/
    └── query_pipeline.py        # QueryPipeline orchestrator
```

## Components

### 1. QueryNormalizer (`query/normalizer.py`)

Vietnamese text normalization cho Wikipedia queries.

**Steps:**
1. Unicode NFC normalization
2. Lowercase
3. Expand abbreviations: TP → thành phố, HN → hà nội, TQ → trung quốc...
4. Normalize whitespace
5. Classify intent từ question patterns
6. Generate expansions

**Intent classification:**
- `definition`: "là gì", "định nghĩa", "có nghĩa"
- `person`: "ai", "là ai", "sinh", "mất"
- `location`: "ở đâu", "nằm ở", "thủ đô"
- `time`: "khi nào", "năm nào"
- `number`: "bao nhiêu", "dân số", "diện tích"
- `history`: "lịch sử", "thành lập"
- `comparison`: "so sánh", "khác nhau"

### 2. QueryRewriter (`query/rewriter.py`)

LLM-based query rewrite producing 3 variants:

- `normalized_query`: cleaned, expanded abbreviations
- `rewrite_query`: semantically expanded version
- `bm25_query`: keyword-optimized for BM25 search

**LLM:** deepseek/deepseek-v4-flash (OpenRouter)

**Fallback:** Nếu LLM fail, dùng simple normalization.

### 3. QueryGuardrails (`query/guardrails.py`)

Input safety checks:

- **Prompt injection**: "ignore previous instructions", "system:", "[INST]"...
- **Unsafe content**: bomb, weapon, hack, illegal...
- **Malformed query**: too short (<3 chars), only special chars, too long (>2000 chars)

### 4. QueryPipeline (`pipelines/query_pipeline.py`)

Orchestrator flow:
```
query → guardrails → normalize → rewrite → ProcessedQuery
```

**Config flags:**
- `enable_rewrite`: bật/tắt LLM rewrite (default: True)
- `enable_guardrails`: bật/tắt safety checks (default: True)

### 5. LLM Client (`indexing/llm_client.py`)

OpenRouter /chat/completions client:

- Bearer token auth từ env var `OPENROUTER_API_KEY`
- Exponential backoff trên 429 rate limit
- `chat()` → text response
- `chat_json()` → parsed JSON response
- `DeterministicTestLLM` cho dev/test

### 6. Vector Store Search (`indexing/vector_store.py`)

Thêm vào Protocol:
- `SearchResult` dataclass: chunk_id, doc_id, text, score, metadata
- `search(query_vector, top_k, filters)` method

## CLI

```powershell
# Normal mode (no LLM, dùng normalization only)
python -m rag_pipeline.main query --question "Thủ đô Việt Nam ở đâu?"

# With LLM rewrite
python -m rag_pipeline.main query --question "Thủ đô Việt Nam ở đâu?" --llm
```

## Output: ProcessedQuery

```python
@dataclass
class ProcessedQuery:
    qid: str
    original_query: str          # "Thủ đô Việt Nam ở đâu?"
    normalized_query: str        # "thủ đô việt nam ở đâu?"
    rewrite_query: str           # "Thủ đô của nước Việt Nam nằm ở đâu?"
    bm25_query: str              # "thủ đô việt nam"
    intent: str                  # "location"
    filters: dict[str, str]      # {}
    risk_flags: list[str]        # []
```

## Config

```python
@dataclass
class LLMConfig:
    model_name: str = "deepseek/deepseek-v4-flash"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    temperature: float = 0.1
    max_tokens: int = 512

@dataclass
class QueryConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    enable_rewrite: bool = True
    enable_guardrails: bool = True
    max_query_length: int = 500
```

## Tests

| File | Tests | Coverage |
|------|-------|----------|
| test_normalizer.py | 14 | Normalization, abbreviation, intent, edge cases |
| test_guardrails.py | 10 | Injection, unsafe, malformed |
| test_query_pipeline.py | 5 | Basic, guardrails, rewrite |
| **Total** | **29** | **48/48 pass** (bao gồm Phase 1 tests) |

## Next: Phase 3 — Retrieval

Dùng `ProcessedQuery` → embed → search Qdrant → trả kết quả có nguồn.
