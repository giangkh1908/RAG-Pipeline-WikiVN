# PLAN.md — RAG Pipeline

Dự án RAG (Retrieval-Augmented Generation) cho Wikipedia tiếng Việt.

---

## Bối cảnh

- Nguồn dữ liệu: Wikipedia tiếng Việt — 1.1M articles (~1.67GB JSONL)
- Vector store: Qdrant chạy trên Docker (ổ D, mount volume)
- Hardware: Intel Ultra 5 245H (14C/14T), 16GB RAM
- Embedding: OpenRouter API → nvidia/llama-nemotron-embed-vl-1b-v2:free (2048-dim)
- LLM: OpenRouter API → meta-llama/llama-3.2-3b-instruct:free

---

## Phase 1: Ingest — HOÀN THÀNH ✅

### Mục tiêu
Nhúng 1.1M Wikipedia articles vào Qdrant.

### Pipeline flow
```
train.jsonl → LocalJsonlReader (+offset index)
    → UVWWikipediaDocumentNormalizer
    → WikipediaArticleCleaner
    → RecursiveChunker (paragraph → sentence → word)
    → OpenRouterEmbeddingClient (4 workers × 500 sub-batch)
    → QdrantVectorStore.batch_upsert (500 points/call)
```

### Benchmark (Intel Ultra 5 245H, 16GB RAM, 4 workers)

| Tỉ lệ | Documents | Thời gian | Tốc độ |
|--------|-----------|-----------|--------|
| 0.01%  | 114       | ~14s      | 8 docs/s |
| 0.05%  | 570       | ~31s      | 18 docs/s |
| 1%     | 11,390    | ~5-7 phút | ~30 docs/s |
| 10%    | 113,900   | ~30-40 phút | ~50 docs/s |
| 100%   | 1,139,000 | ~5-7 giờ  | ~50 docs/s |

### CLI
```powershell
python -m rag_pipeline.main ingest [--sample 0.1] [--clear]
```

---

## Phase 2: Query Processing — HOÀN THÀNH ✅

### Mục tiêu
Xử lý user query trước khi retrieval: normalize → guardrails → rewrite.

### Pipeline flow
```
user query
    → QueryGuardrails (prompt injection, unsafe content)
    → QueryNormalizer (Vietnamese normalization, intent classification)
    → QueryRewriter (LLM → normalized/rewrite/bm25 variants)
    → ProcessedQuery
```

### CLI
```powershell
python -m rag_pipeline.main query --question "Thủ đô Việt Nam ở đâu?" [--llm]
```

---

## Phase 3: Retrieval (Hybrid Search + Re-ranking) — HOÀN THÀNH ✅

### Mục tiêu
Từ ProcessedQuery → hybrid search (dense + BM25) → RRF fusion → BGE re-rank → top-k results.

### Pipeline flow
```
ProcessedQuery
    ├─ Dense search: embed rewrite_query → Qdrant top_k=50
    ├─ BM25 search: bm25_query → BM25 index top_k=50
    └─ RRF fusion: merge 2 kết quả → top_k=20
    → BGE Re-ranker: re-rank top 20 → top 5
    → RetrievalResult (passages + context)
```

### Sẽ làm

**3.1. BM25 Index**
- Library: `rank_bm25` hoặc `gensim`
- Tokenizer: Vietnamese word segmentation (underthesea hoặc pyvi)
- Xây dựng index từ 1.1M documents
- Lưu index vào disk (pickle hoặc gensim format)
- Update index khi có documents mới

**3.2. Dense Vector Search**
- Embed `rewrite_query` bằng OpenRouter embedding (2048-dim)
- Search Qdrant: cosine similarity, top_k=50
- Lọc theo filters nếu có (intent, metadata)

**3.3. BM25 Keyword Search**
- Tokenize `bm25_query` bằng Vietnamese tokenizer
- Search BM25 index, top_k=50
- Trả về (doc_id, score) pairs

**3.4. RRF Fusion (Reciprocal Rank Fusion)**
- Formula: `score(d) = Σ 1/(k + rank_i(d))` với k=60
- Merge kết quả dense + BM25
- Sort theo RRF score descending
- Lấy top 20 passages

**3.5. BGE Re-ranking**
- Model: `BAAI/bge-reranker-v2-m3` (cross-encoder, hỗ trợ đa ngôn ngữ)
- Input: (query, passage) pairs cho top 20
- Output: relevance scores
- Sort theo score descending → lấy top 5
- Có thể chạy local (transformers) hoặc API

**3.6. Context Assembly**
- Lấy top 5 passages
- Gắn source_url, title, chunk_index
- Format thành context string cho Phase 4

### Output
```python
@dataclass
class RetrievalResult:
    query: ProcessedQuery
    passages: list[Passage]       # top 5 passages với score
    context: str                   # assembled context string

@dataclass
class Passage:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_url: str
    dense_score: float             # cosine similarity score
    bm25_score: float              # BM25 score
    rrf_score: float               # RRF fusion score
    rerank_score: float            # BGE re-rank score
    rank: int                      # final rank (1-5)
```

### CLI
```powershell
python -m rag_pipeline.main search --question "Thủ đô Việt Nam ở đâu?"
```

---

## Phase 4: Orchestration + Generation — CHƯA LÀM

### Mục tiêu
Orchestrate toàn bộ pipeline (query → retrieve → generate) + output guardrails cho generated answer.

### Pipeline flow
```
User question
    → [Phase 2] Query Processing → ProcessedQuery
    → [Phase 3] Hybrid Retrieval → RetrievalResult
    → [Phase 4] Generation
        → Build prompt (context + question)
        → LLM generate (OpenRouter /chat/completions)
        → Post-process (format citations, extract answer)
        → Output Guardrails (hallucination check, safety, quality)
        → Final AnswerResult
```

### Sẽ làm

**4.1. Orchestration**
- `RAGPipeline` class: orchestrate Phase 2 → 3 → 4
- Config-driven: bật/tắt từng stage
- Error handling: fallback khi LLM fail, khi retrieval empty
- Logging: trace mỗi stage (latency, input, output)

**4.2. Prompt Engineering**
```
Bạn là trợ lý tìm kiếm Wikipedia tiếng Việt. Dựa vào thông tin dưới đây, trả lời câu hỏi.

Thông tin:
{context}

Câu hỏi: {question}

Quy tắc:
1. Chỉ trả lời dựa trên thông tin được cung cấp
2. Nếu không tìm thấy thông tin, nói "Không tìm thấy thông tin trong cơ sở dữ liệu."
3. Trả lời ngắn gọn, chính xác
4. Đánh dấu nguồn tham khảo: [1], [2]...
```

**4.3. Citation Injection**
- Mỗi passage có source_url → đánh dấu [1], [2]... trong answer
- Footer: list nguồn tham khảo
- Verify: citation có trỏ đúng passage không

**4.4. Output Guardrails**
- **Hallucination check**: answer có dựa trên context không? (so sánh với passages)
- **Safety filter**: answer có chứa nội dung unsafe không?
- **Quality check**: answer có quá ngắn/ngắn dài không? Có trả lời đúng câu hỏi không?
- **Confidence scoring**: dựa trên retrieval scores + LLM confidence

**4.5. Output**
```python
@dataclass
class AnswerResult:
    answer: str                    # generated answer
    citations: list[Citation]      # [1] source_url, [2] source_url...
    confidence: float              # 0.0 - 1.0
    retrieval_result: RetrievalResult
    guardrail_flags: list[str]     # hallucination, unsafe, low_quality...
    metadata: dict[str, Any]       # latency, token usage, model used

@dataclass
class Citation:
    index: int
    title: str
    source_url: str
    snippet: str
```

### CLI
```powershell
# Full pipeline: question → answer
python -m rag_pipeline.main ask --question "Thủ đô Việt Nam ở đâu?"

# Verbose: show retrieval details
python -m rag_pipeline.main ask --question "..." --verbose
```

---

## Phase 5: Eval + Monitoring — CHƯA LÀM

### Mục tiêu
Đo lường chất lượng RAG pipeline: tracing, logging, alerting, eval metrics.

### Components

**5.1. Tracing**
- Trace mỗi request qua các stage: query → normalize → rewrite → retrieve → generate
- Library: LangSmith hoặc OpenTelemetry
- Lưu trace vào local DB hoặc external service
- Metadata: latency per stage, token usage, model used

**5.2. Logging**
- Structured logging (JSON format)
- Log levels: DEBUG (dev), INFO (prod), WARNING (slow/error)
- Log mỗi step: input, output, latency, error
- Rotation: daily hoặc theo size

**5.3. Alerting**
- Threshold-based alerts:
  - Error rate > 5% → alert
  - P95 latency > 10s → alert
  - Low confidence answers (< 0.3) → alert
  - Empty results rate > 10% → alert
- Channel: log file, webhook, email (optional)

**5.4. Eval Metrics**

Dùng **RAGAS** hoặc **TruLens** để đánh giá:

| Metric | Mô tả | Target |
|--------|--------|--------|
| **Faithfulness** | Answer có dựa trên context không? (hallucination detection) | ≥ 0.8 |
| **Answer Relevance** | Answer có liên quan đến question không? | ≥ 0.7 |
| **Context Precision** | Retrieved context có chính xác không? (noise ratio) | ≥ 0.7 |
| **Context Recall** | Retrieved context có đầy đủ không? (coverage) | ≥ 0.6 |

**Eval Flow:**
```
Eval dataset (100-200 mẫu)
    → Chạy full pipeline cho mỗi mẫu
    → Tính metrics bằng RAGAS/TruLens
    → Report: per-metric scores, per-query breakdown
    → So sánh giữa các version (A/B test)
```

**Eval Dataset:**
- 100-200 mẫu từ train.csv (question + ground_truth_answer)
- Chia thành: easy (50%), medium (35%), hard (15%)
- Mỗi mẫu có: question, expected_answer, expected_sources

**5.5. Dashboard (optional)**
- Grafana hoặc Streamlit dashboard
- Metrics: avg latency, error rate, faithfulness trend, answer quality
- Real-time monitoring during ingest/query

### CLI
```powershell
# Chạy eval
python -m rag_pipeline.main eval --dataset document/eval.csv --output report.json

# Xem metrics
python -m rag_pipeline.main metrics --last-n 100
```

### Config
```python
@dataclass
class EvalConfig:
    eval_dataset_path: Path = Path("document/eval.csv")
    metrics: list[str] = field(default_factory=lambda: [
        "faithfulness", "answer_relevance", "context_precision", "context_recall"
    ])
    faithfulness_threshold: float = 0.8
    answer_relevance_threshold: float = 0.7
    context_precision_threshold: float = 0.7
    context_recall_threshold: float = 0.6

@dataclass
class MonitoringConfig:
    enable_tracing: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    alert_error_rate_threshold: float = 0.05
    alert_latency_p95_threshold: float = 10.0
    alert_confidence_threshold: float = 0.3
```

---

## Tổng kết

| Phase | Trạng thái | Mô tả |
|-------|-----------|-------|
| 1. Ingest | ✅ Hoàn thành | 1.1M docs → Qdrant |
| 2. Query Processing | ✅ Hoàn thành | Normalize → guardrails → rewrite |
| 3. Retrieval | ✅ Hoàn thành | Hybrid (dense + BM25) → RRF → BGE re-rank |
| 4. Orchestration + Generation | 🔲 Chưa làm | RAG pipeline → LLM generate → output guardrails |
| 5. Eval + Monitoring | 🔲 Chưa làm | RAGAS/TruLens + tracing + alerting |
