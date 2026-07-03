# RAG Pipeline — Vietnamese Wikipedia

Production-first RAG (Retrieval-Augmented Generation) pipeline cho Wikipedia tiếng Việt (1.1M articles).

## Features

- **Hybrid Search**: Dense vector (Qdrant) + BM25 keyword search → RRF fusion
- **Re-ranking**: Cohere Rerank v3.5 (multilingual, hỗ trợ tiếng Việt)
- **Streaming**: Real-time token streaming (TTFT ~2-3s)
- **LangSmith Tracing**: Auto-enabled tracing cho debugging/monitoring
- **RAGAS Evaluation**: 4 quality metrics + latency metrics (TTFT, P50/P90/P99)
- **Output Guardrails**: Hallucination detection, safety check, quality check

## Architecture

```
User Question
     │
     ▼
┌─────────────────┐
│  Query Pipeline  │  Phase 2: guardrails → normalize → rewrite
└────────┬────────┘
         │ ProcessedQuery
         ▼
┌─────────────────┐
│ Retrieval Pipeline│  Phase 3: dense + BM25 → RRF → Cohere rerank
└────────┬────────┘
         │ RetrievalResult
         ▼
┌─────────────────┐
│ Prompt Builder   │  Build system + user messages
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Answer Generator │  LLM → parse JSON → AnswerResult + Citations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Output Guardrails │  Hallucination + Safety + Quality check
└────────┬────────┘
         │
         ▼
    Final Answer + Citations + Confidence
```

## Quick Start

### 1. Setup

```bash
# Clone repository
git clone <repo-url>
cd RAG

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[indexing,monitoring,eval]"
```

### 2. Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
# Required
OPENROUTER_API_KEY=sk-or-v1-xxx

# Qdrant (default: localhost:6333)
QDRANT_URL=http://localhost:6333

# Optional: Cohere re-ranking
COHERE_API_KEY=xxx

# Optional: LangSmith tracing
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_xxx
LANGSMITH_PROJECT=rag-pipeline
LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com
```

### 3. Start Qdrant

```bash
docker-compose up -d
```

### 4. Ingest Data

```bash
# Full dataset (1.1M docs, ~5-7 hours)
python -m rag_pipeline.main ingest

# Sample (for testing)
python -m rag_pipeline.main ingest --sample 0.1
```

### 5. Ask Questions

```bash
# Standard mode
python -m rag_pipeline.main ask --question "Wikipedia là gì?" --text

# Streaming mode (tokens appear in real-time)
python -m rag_pipeline.main ask --question "Wikipedia là gì?" --text --stream
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ingest` | Ingest documents into Qdrant |
| `query` | Process a query (normalize + rewrite) |
| `search` | Search documents (hybrid search) |
| `ask` | Full RAG pipeline (question → answer) |
| `eval` | Run RAGAS evaluation |

### Common Options

```bash
# ask command
--question TEXT    # Question to ask
--text             # User-friendly output (answer + 1 source)
--stream           # Stream tokens in real-time
--no-qdrant        # Use InMemory instead of Qdrant
--rerank           # Use Cohere re-ranker
--no-llm           # Disable LLM query rewrite

# eval command
--dataset PATH     # Eval dataset CSV (default: documents/eval.csv)
--output PATH      # Output report path (default: eval_report.json)
--limit N          # Max samples to evaluate (default: 50)
```

## Python API

```python
from rag_pipeline.main import ask, build_ask_pipeline

# Simple usage
result = ask("Wikipedia là gì?")
print(result.answer)
print(result.citations)
print(result.confidence)

# Streaming mode
pipeline = build_ask_pipeline()
processed = pipeline._run_query_processing("Wikipedia là gì?")
retrieval = pipeline._run_retrieval(processed)

chunk_gen, build_result = pipeline.answer_generator.generate_stream(retrieval)

for chunk in chunk_gen:
    print(chunk, end="", flush=True)

result = build_result(full_text)
```

## Evaluation

### Quality Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| Faithfulness | Answer dựa trên context? (hallucination detection) | ≥ 0.8 |
| Answer Relevancy | Answer liên quan đến question? | ≥ 0.7 |
| Context Precision | Retrieved context chính xác? | ≥ 0.7 |
| Context Recall | Retrieved context đầy đủ? | ≥ 0.6 |

### Latency Metrics

| Metric | Description |
|--------|-------------|
| TTFT (Time to First Token) | Thời gian chờ token đầu tiên |
| TTFT P50/P90/P99 | Percentiles cho TTFT |
| Total P50/P90/P99 | Percentiles cho tổng thời gian |
| Query Processing | Thời gian xử lý query |
| Retrieval | Thời gian tìm kiếm |
| Generation | Thời gian sinh câu trả lời |

### Run Evaluation

```bash
# Run eval with default dataset
python -m rag_pipeline.main eval

# Run with custom settings
python -m rag_pipeline.main eval --dataset documents/eval.csv --limit 10 --output eval_report.json
```

### Sample Output

```
📊 Evaluation Results:
============================================================

  Quality Metrics:
  --------------------------------------------------------
    faithfulness.................. 0.8500  (threshold: 0.8) ✅
    answer_relevancy.............. 0.7800  (threshold: 0.7) ✅
    context_precision............. 0.7200  (threshold: 0.7) ✅
    context_recall................ 0.6500  (threshold: 0.6) ✅

  Latency Metrics:
  --------------------------------------------------------
    TTFT (P50)..................     320ms
    TTFT (P90)..................     450ms
    TTFT (avg)..................     340ms
    Total (P50)..................    2200ms
    Total (P90)..................    2800ms
    Total (avg)..................    2350ms
    Query Processing (avg).......     650ms
    Retrieval (avg)..............     190ms
    Generation (avg).............    1500ms

============================================================
  Overall: ✅ PASS
  Samples: 5
```

## Monitoring

### LangSmith Tracing

Tracing auto-enabled khi `LANGSMITH_TRACING_V2=true` trong `.env`.

1. Vào https://smith.langchain.com
2. Chọn project `rag-pipeline`
3. Xem traces cho mỗi `ask` call

## Project Structure

```
RAG/
├── src/rag_pipeline/
│   ├── __init__.py
│   ├── main.py                 # CLI + factory functions
│   ├── config.py               # All configs (dataclass)
│   ├── models.py               # Data models (dataclass)
│   │
│   ├── query/                  # Phase 2: Query Processing
│   │   ├── guardrails.py       # Prompt injection detection
│   │   ├── normalizer.py       # Vietnamese normalization
│   │   └── rewriter.py         # LLM query rewrite
│   │
│   ├── indexing/               # Phase 1+3: Indexing & Search
│   │   ├── bm25_index.py       # BM25 keyword index
│   │   ├── embedder.py         # OpenRouter embedding client
│   │   ├── llm_client.py       # OpenRouter LLM client (streaming)
│   │   ├── reranker.py         # Cohere + BGE reranker
│   │   └── vector_store.py     # Qdrant + InMemory vector store
│   │
│   ├── transform/              # Phase 1: Data transformation
│   │   ├── chunker.py          # Recursive chunking
│   │   └── cleaner.py          # Wikipedia article cleaner
│   │
│   ├── ingest/                 # Phase 1: Data ingestion
│   │   ├── dataset.py          # Dataset readers
│   │   └── normalize.py        # Document normalization
│   │
│   ├── pipelines/              # Orchestration
│   │   ├── answer_pipeline.py  # Phase 4: Full RAG pipeline
│   │   ├── ingest_pipeline.py  # Phase 1: Ingest pipeline
│   │   ├── query_pipeline.py   # Phase 2: Query pipeline
│   │   └── retrieval_pipeline.py # Phase 3: Retrieval pipeline
│   │
│   ├── generation/             # Phase 4: Generation
│   │   ├── prompt_builder.py   # Build LLM prompts
│   │   ├── answer_generator.py # Generate answers (streaming)
│   │   └── output_guardrails.py # Hallucination/safety/quality
│   │
│   └── eval/                   # Phase 5: Evaluation
│       ├── runner.py           # RAGAS eval runner + latency
│       └── report.py           # Eval report (JSON + Markdown)
│
├── tests/                      # 96 tests
├── documents/                  # Eval dataset
├── docs/                       # Documentation
├── pyproject.toml              # Project config
└── README.md                   # This file
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Vector Store | Qdrant (Docker) |
| Embedding | OpenRouter → nvidia/llama-nemotron-embed-vl-1b-v2:free (2048-dim) |
| LLM | OpenRouter → deepseek/deepseek-v4-flash |
| Re-ranking | Cohere Rerank v3.5 |
| BM25 | rank-bm25 + pyvi (Vietnamese tokenizer) |
| Tracing | LangSmith |
| Evaluation | RAGAS + LiteLLM |
| HTTP Client | httpx |
| Testing | pytest |

## Performance

### Latency Breakdown

| Step | Time | % |
|------|------|---|
| Query Processing | ~650ms | 27% |
| Retrieval | ~190ms | 7% |
| Generation | ~1500ms | 66% |
| **Total** | ~2350ms | 100% |

### Streaming Improvement

| Mode | TTFT | Total |
|------|------|-------|
| Non-streaming | - | ~17s |
| Streaming | ~2-3s | ~17s |

Streaming không giảm total latency, nhưng giảm perceived latency (user thấy token ngay).

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_generation.py -v

# Run with coverage
python -m pytest tests/ --cov=rag_pipeline
```

96 tests covering:
- Ingest pipeline (5 tests)
- Query processing (6 tests)
- Retrieval pipeline (7 tests)
- Generation (18 tests)
- Evaluation (8 tests)
- LangSmith logging (4 tests)

## Documentation

- [PLAN.md](PLAN.md) — Project plan & progress
- [docs/generation.md](docs/generation.md) — Phase 4: Generation pipeline
- [docs/eval.md](docs/eval.md) — Phase 5: Evaluation metrics

## License

MIT
