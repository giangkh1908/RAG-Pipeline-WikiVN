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
Từ ProcessedQuery → hybrid search (dense + BM25) → RRF fusion → Cohere re-rank → top-k results.

### Pipeline flow
```
ProcessedQuery
    ├─ Dense search: embed rewrite_query → Qdrant top_k=50
    ├─ BM25 search: bm25_query → BM25 index top_k=50
    └─ RRF fusion: merge 2 kết quả → top_k=20
    → Cohere Re-ranker: re-rank top 20 → top 5
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

**3.5. Cohere Re-ranking**
- API: Cohere Rerank API v2 (`https://api.cohere.com/v2/rerank`)
- Model: `rerank-v3.5` (multilingual, hỗ trợ tiếng Việt)
- Free tier: 100 search units/tháng
- Input: query + documents (top 20 passages)
- Output: relevance scores [0, 1]
- Sort theo score descending → lấy top 5

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
    rerank_score: float            # Cohere re-rank score
    rank: int                      # final rank (1-5)
```

### CLI
```powershell
python -m rag_pipeline.main search --question "Thủ đô Việt Nam ở đâu?"
```

---

## Phase 4: Orchestration + Generation — HOÀN THÀNH ✅

### Mục tiêu
Orchestrate toàn bộ pipeline (query → retrieve → generate) + output guardrails cho generated answer.

### Pipeline flow
```
User question
    → [Phase 2] Query Processing → ProcessedQuery
    → [Phase 3] Hybrid Retrieval → RetrievalResult
    → [Phase 4] Generation
        → PromptBuilder (system + user messages)
        → LLM generate (OpenRouter /chat/completions)
        → AnswerGenerator (parse JSON → AnswerResult + Citations)
        → OutputGuardrails (hallucination, safety, quality)
        → Final AnswerResult
```

### Đã làm

**4.1. Orchestration**
- `AnswerPipeline` class: orchestrate Phase 2 → 3 → 4
- Config-driven: GenerationConfig, OutputGuardrailsConfig
- Error handling: fallback khi JSON parse fail → wrap raw text

**4.2. Prompt Engineering**
- `PromptBuilder` — build system/user messages
- System: hướng dẫn trả lời tiếng Việt, trích dẫn [1][2], trả về JSON
- User: passages đánh số + câu hỏi

**4.3. Citation Injection**
- `AnswerGenerator` — gọi LLM, parse JSON response
- Map `source_index` từ LLM về passage gốc
- Tạo `Citation` objects với claim, chunk_id, doc_id, title, source_url

**4.4. Output Guardrails**
- **Hallucination check**: verify claims backed by passages
- **Safety check**: detect unsafe content
- **Quality check**: answer length, min_citations
- Confidence giảm 0.2 mỗi flag

**4.5. Streaming Support**
- `OpenRouterLLMClient.stream()` — SSE streaming từ OpenRouter API
- `AnswerGenerator.generate_stream()` — yield chunks + build_result function
- CLI `--stream` flag — print tokens real-time (giảm perceived latency)
- TTFT (Time to First Token) ~2-3s thay vì chờ 17s full response

**4.6. Output**
```python
@dataclass
class AnswerResult:
    question: str                   # Original question
    answer: str                     # LLM-generated answer
    citations: list[Citation]       # Source citations
    confidence: float               # Overall confidence (0-1)
    passages_used: int              # Number of passages used
    metadata: dict[str, Any]        # Guardrail flags, parse mode, etc.

@dataclass
class Citation:
    claim: str          # Claim in the answer
    chunk_id: str       # Source passage ID
    doc_id: str         # Source document ID
    title: str          # Article title
    source_url: str     # Wikipedia URL
    confidence: float   # Citation confidence (0-1)
```

### CLI
```powershell
# Full pipeline: question → answer
python -m rag_pipeline.main ask --question "Thủ đô Việt Nam ở đâu?"

# User-friendly output (answer + 1 source)
python -m rag_pipeline.main ask --question "..." --text

# Streaming mode (tokens appear in real-time)
python -m rag_pipeline.main ask --question "..." --text --stream
```

---

## Phase 5: Eval + Monitoring — HOÀN THÀNH ✅

### Mục tiêu
Đo lường chất lượng RAG pipeline: LangSmith tracing + RAGAS eval metrics + latency metrics.

### Đã làm

**5.1. Tracing — LangSmith**
- Tích hợp LangSmith tracing (auto-enabled khi `LANGSMITH_TRACING_V2=true` trong `.env`)
- Trace từng stage: query_processing → retrieval → generation → output_guardrails
- Dashboard: https://smith.langchain.com

**5.2. Quality Metrics — RAGAS**

| Metric | Mô tả | Target | Cần LLM? |
|--------|--------|--------|----------|
| **Faithfulness** | Answer có dựa trên context không? | ≥ 0.8 | ✅ |
| **Answer Relevancy** | Answer có liên quan đến question không? | ≥ 0.7 | ✅ |
| **Context Precision** | Retrieved context có chính xác không? | ≥ 0.7 | ❌ |
| **Context Recall** | Retrieved context có đầy đủ không? | ≥ 0.6 | ❌ |

**5.3. Latency Metrics**

| Metric | Mô tả |
|--------|--------|
| **TTFT (Time to First Token)** | Thời gian chờ token đầu tiên |
| **TTFT P50/P90/P99** | Percentiles cho TTFT |
| **Total P50/P90/P99** | Percentiles cho tổng thời gian |
| **Query Processing** | Thời gian xử lý query |
| **Retrieval** | Thời gian tìm kiếm |
| **Generation** | Thời gian sinh câu trả lời |

**Eval Flow:**
```
documents/eval.csv
    → EvalRunner.load_dataset()
    → AnswerPipeline (query + retrieval + generation) cho mỗi sample
    → Measure TTFT + latency per step
    → RAGAS.evaluate() với 4 metrics
    → EvalReport (JSON + Markdown) với quality + latency
```

### CLI
```powershell
# Chạy eval
python -m rag_pipeline.main eval --dataset documents/eval.csv --limit 50 --output eval_report.json
```

### Config
```python
@dataclass
class EvalConfig:
    eval_dataset_path: Path = Path("documents/eval.csv")
    llm_model: str = "deepseek/deepseek-v4-flash"
    faithfulness_threshold: float = 0.8
    answer_relevance_threshold: float = 0.7
    context_precision_threshold: float = 0.7
    context_recall_threshold: float = 0.6
```

---

## Tổng kết

| Phase | Trạng thái | Mô tả |
|-------|-----------|-------|
| 1. Ingest | ✅ Hoàn thành | 1.1M docs → Qdrant |
| 2. Query Processing | ✅ Hoàn thành | Normalize → guardrails → rewrite |
| 3. Retrieval | ✅ Hoàn thành | Hybrid (dense + BM25) → RRF → Cohere re-rank |
| 4. Orchestration + Generation | ✅ Hoàn thành | PromptBuilder + AnswerGenerator + OutputGuardrails + Streaming + CLI `ask` |
| 5. Eval + Monitoring | ✅ Hoàn thành | LangSmith tracing + RAGAS eval (4 metrics) + Latency metrics (TTFT, P50/P90/P99) |

## Test Coverage

| Category | Tests | Description |
|----------|-------|-------------|
| Ingest | 5 | Document loading, chunking, normalization |
| Query | 6 | Normalizer, guardrails, query pipeline |
| Retrieval | 7 | RRF fusion, retrieval pipeline, vector store |
| Generation | 18 | Prompt builder, answer generator, output guardrails, pipeline |
| Eval | 8 | EvalReport, EvalConfig, dataset loading |
| Logging | 4 | LangSmith config, tracing integration |
| **Total** | **96** | All tests pass ✅ |
