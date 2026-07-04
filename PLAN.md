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

## Phase 6: FastAPI Backend — CHƯA LÀM

### Mục tiêu
Tạo REST API + SSE streaming để frontend gọi RAG pipeline.

### Architecture

```
Frontend (React)
     │
     │  fetch + ReadableStream (SSE)
     ▼
┌─────────────────┐
│   FastAPI        │
│   :8000          │
├─────────────────┤
│ POST /api/chat   │ → pipeline.ask() → AnswerResult (JSON)
│ GET /api/chat/stream │ → StreamingResponse → tokens (SSE)
│ GET /api/health  │ → status check
│ POST /api/eval   │ → runner.run() → EvalReport
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Pipeline    │  (Phase 2→3→4, giữ nguyên code cũ)
└─────────────────┘
```

### Streaming Flow (SSE + ReadableStream)

```
Client                          Server
  │                               │
  │  GET /api/chat/stream?q=...   │
  │──────────────────────────────▶│
  │                               │ pipeline.generate_stream()
  │  data: "Xin"                  │
  │◀──────────────────────────────│
  │  data: "chào"                 │
  │◀──────────────────────────────│
  │  data: "bạn"                  │
  │◀──────────────────────────────│
  │  data: [DONE]                 │
  │◀──────────────────────────────│
  │                               │
```

### Files cần tạo

```
src/rag_pipeline/api/
├── __init__.py
├── app.py              # FastAPI app, CORS, lifespan
├── routes/
│   ├── __init__.py
│   ├── chat.py         # POST /api/chat + GET /api/chat/stream (SSE)
│   ├── eval.py         # POST /api/eval
│   └── health.py       # GET /api/health
└── schemas.py          # Pydantic request/response models
```

### Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| `POST` | `/api/chat` | Gửi câu hỏi, nhận answer + citations (JSON) |
| `GET` | `/api/chat/stream` | SSE streaming tokens real-time |
| `GET` | `/api/health` | Health check + Qdrant status |
| `POST` | `/api/eval` | Chạy evaluation |

### Streaming Implementation

```python
# Backend (FastAPI)
from fastapi.responses import StreamingResponse

@app.get("/api/chat/stream")
async def chat_stream(question: str):
    async def generate():
        pipeline = get_pipeline()
        processed = pipeline._run_query_processing(question)
        retrieval = pipeline._run_retrieval(processed)
        chunk_gen, build_result = pipeline.answer_generator.generate_stream(retrieval)

        for chunk in chunk_gen:
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        result = build_result(full_text)
        yield f"data: {json.dumps({'type': 'done', 'citations': [...]})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

```javascript
// Frontend (React)
const response = await fetch(`/api/chat/stream?question=${question}`);
const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const text = decoder.decode(value);
  // Parse SSE format: "data: {...}\n\n"
  const lines = text.split('\n').filter(l => l.startsWith('data: '));

  for (const line of lines) {
    const data = JSON.parse(line.slice(6));
    if (data.type === 'token') {
      appendToken(data.content);  // Hiển thị ngay
    } else if (data.type === 'done') {
      showCitations(data.citations);
    }
  }
}
```

### Schemas (Pydantic)

```python
# Request
class ChatRequest(BaseModel):
    question: str
    use_reranker: bool = False
    use_llm: bool = True

# Response (non-streaming)
class CitationResponse(BaseModel):
    claim: str
    title: str
    source_url: str
    confidence: float

class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    confidence: float
    latency_ms: float

# SSE Stream format
# data: {"type": "token", "content": "Xin"}
# data: {"type": "token", "content": "chào"}
# data: {"type": "done", "citations": [...]}
```

### Dependencies

```toml
[project.optional-dependencies]
api = ["fastapi>=0.115.0", "uvicorn[standard]>=0.34.0"]
```

### CLI

```powershell
# Chạy API server
python -m rag_pipeline.api.app
# Hoặc
uvicorn rag_pipeline.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Tasks

- [x] Tạo `api/schemas.py` — Pydantic models
- [x] Tạo `api/app.py` — FastAPI app + CORS + lifespan
- [x] Tạo `api/routes/health.py` — Health check
- [x] Tạo `api/routes/chat.py` — POST /api/chat + SSE stream
- [x] Tạo `api/routes/eval.py` — POST /api/eval
- [x] Tests cho API endpoints (13 passed, 2 skipped)
- [x] Update pyproject.toml

---

## Phase 7: React Frontend — ✅ HOÀN THÀNH

### Mục tiêu
Tạo chat UI kết nối FastAPI backend qua SSE + ReadableStream.

### Architecture

```
┌─────────────────────────────────────────┐
│              React App                   │
│              (Vite + TypeScript)         │
├─────────────────────────────────────────┤
│  ┌───────────┐  ┌───────────────────┐   │
│  │  Sidebar   │  │    ChatBox        │   │
│  │  - History │  │  - Messages       │   │
│  │  - Settings│  │  - Input          │   │
│  │            │  │  - Citations      │   │
│  └───────────┘  └───────────────────┘   │
├─────────────────────────────────────────┤
│  hooks/useChat.ts  (SSE streaming)      │
│  api/client.ts     (fetch + ReadableStream) │
└─────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | React 19 + TypeScript |
| Build tool | Vite 6 |
| Styling | Tailwind CSS v4 |
| HTTP client | fetch API + ReadableStream (built-in) |
| Icons | Lucide React |

### Streaming Logic (useChat hook)

```typescript
// hooks/useChat.ts
const sendMessage = async (question: string) => {
  // Add user message
  setMessages(prev => [...prev, { role: 'user', content: question }]);

  // Start streaming
  const response = await fetch(`/api/chat/stream?question=${encodeURIComponent(question)}`);
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  let botMessage = '';
  setMessages(prev => [...prev, { role: 'bot', content: '', streaming: true }]);

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const lines = text.split('\n').filter(l => l.startsWith('data: '));

    for (const line of lines) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'token') {
        botMessage += data.content;
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'bot', content: botMessage, streaming: true };
          return updated;
        });
      } else if (data.type === 'done') {
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'bot',
            content: botMessage,
            citations: data.citations,
            streaming: false
          };
          return updated;
        });
      }
    }
  }
};
```

### Files cần tạo

```
frontend/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
├── index.html
├── public/
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── index.css                 # Tailwind imports
    ├── components/
    │   ├── ChatBox.tsx           # Chat container + messages list
    │   ├── MessageBubble.tsx     # Single message (user/bot)
    │   ├── ChatInput.tsx         # Input + send button
    │   ├── CitationCard.tsx      # Citation display
    │   ├── Sidebar.tsx           # History + settings
    │   └── Layout.tsx            # Main layout
    ├── hooks/
    │   └── useChat.ts            # Chat state + SSE streaming logic
    ├── api/
    │   └── client.ts             # fetch + ReadableStream wrapper
    └── types/
        └── index.ts              # TypeScript types
```

### Features

**ChatBox:**
- Hiển thị messages (user + bot)
- Streaming text (token by token) via ReadableStream
- Loading indicator khi chờ response
- Auto scroll xuống cuối

**MessageBubble:**
- User message: right-aligned, blue
- Bot message: left-aligned, gray
- Markdown rendering
- Citation links clickable

**CitationCard:**
- Hiển thị source URL
- Claim text
- Confidence score

**Sidebar:**
- Chat history (localStorage)
- Settings (model, reranker toggle)
- Clear history

### Tasks

- [x] Init Vite + React 19 + TypeScript project
- [x] Cài Tailwind CSS v4 (@tailwindcss/vite)
- [x] Tạo `types/index.ts` — TypeScript types matching backend schemas
- [x] Tạo `api/client.ts` — fetch + ReadableStream wrapper
- [x] Tạo `hooks/useChat.ts` — Chat state + SSE streaming
- [x] Tạo `components/ChatInput.tsx` — Input component
- [x] Tạo `components/MessageBubble.tsx` — Message display
- [x] Tạo `components/CitationCard.tsx` — Citation display
- [x] Tạo `App.tsx` — Root component (gộp ChatBox + Layout)
- [x] Style với Tailwind CSS
- [x] Vite proxy → backend (dev mode)
- [x] FastAPI serve frontend/dist (production)
- [x] Build pass, TypeScript no errors
- [x] Responsive design (mobile + desktop)
- [x] ChatGPT-style UI (auto-expand input, suggestions, numbered citations)

---

## Phase 8: Docker + Deploy — HOÀN THÀNH ✅

### Mục tiêu
Dockerize toàn bộ stack để deploy lên VPS.

### Architecture

```
┌─────────────────────────────────────────────────┐
│                 Docker Compose                  │
│                                                 │
│  ┌──────────────────┐  ┌──────────┐             │
│  │     API + UI     │  │  Qdrant  │             │
│  │  (FastAPI + dist)│  │  :6333   │             │
│  │      :8000       │  │          │             │
│  └────────┬─────────┘  └────┬─────┘             │
│           │                 │                   │
│           └─────────────────┘                   │
│                                                 │
│  Frontend build → served by FastAPI static      │
└─────────────────────────────────────────────────┘
```

### Files

```
Dockerfile                    # Multi-stage: Node build + Python serve
.dockerignore                 # Exclude unnecessary files
docker-compose.yml            # Full stack (Qdrant + API + Frontend)
```

### docker-compose.yml

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.13.6
    container_name: rag-qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 3

  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-api
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      qdrant:
        condition: service_healthy
    restart: unless-stopped

volumes:
  qdrant_storage:
```

### Deploy Flow

```bash
# 1. Clone repo
git clone <repo> && cd RAG

# 2. Tạo .env
cp .env.example .env
# Edit .env với API keys

# 3. Build + start
docker-compose up --build -d

# 4. Truy cập
# http://localhost:8000
```

### Tasks

- [x] Tạo `Dockerfile` (multi-stage: Node build + Python serve)
- [x] Tạo `.dockerignore`
- [x] Cập nhật `docker-compose.yml` (Qdrant + API)
- [x] Cập nhật `README.md` với Docker instructions

---

## Tổng kết

| Phase | Trạng thái | Mô tả |
|-------|-----------|-------|
| 1. Ingest | ✅ Hoàn thành | 1.1M docs → Qdrant |
| 2. Query Processing | ✅ Hoàn thành | Normalize → guardrails → rewrite |
| 3. Retrieval | ✅ Hoàn thành | Hybrid (dense + BM25) → RRF → Cohere re-rank |
| 4. Orchestration + Generation | ✅ Hoàn thành | PromptBuilder + AnswerGenerator + OutputGuardrails + Streaming + CLI `ask` |
| 5. Eval + Monitoring | ✅ Hoàn thành | LangSmith tracing + RAGAS eval (4 metrics) + Latency metrics (TTFT, P50/P90/P99) |
| 6. FastAPI Backend | ✅ Hoàn thành | REST API + SSE streaming (ReadableStream) |
| 7. React Frontend | ✅ Hoàn thành | Chat UI + SSE streaming + citations + Responsive |
| 8. Docker + Deploy | ✅ Hoàn thành | Docker Compose + GitHub Actions CD + GHCR |

## Tech Stack Summary

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + SSE (Server-Sent Events) |
| Frontend | React 19 + Vite 8 + TypeScript + Tailwind v4 |
| Streaming | SSE + ReadableStream (built-in browser API) |
| Vector Store | Qdrant (Docker) |
| LLM | OpenRouter (DeepSeek) |
| Embedding | OpenRouter (NVIDIA Nemotron) |
| Re-ranking | Cohere Rerank v3.5 |
| Tracing | LangSmith |
| Evaluation | RAGAS |

## Test Coverage

| Category | Tests | Description |
|----------|-------|-------------|
| Ingest | 5 | Document loading, chunking, normalization |
| Query | 6 | Normalizer, guardrails, query pipeline |
| Retrieval | 7 | RRF fusion, retrieval pipeline, vector store |
| Generation | 18 | Prompt builder, answer generator, output guardrails, pipeline |
| Eval | 8 | EvalReport, EvalConfig, dataset loading |
| Logging | 4 | LangSmith config, tracing integration |
| API | 13 | Health, root, chat, stream, eval endpoints |
| **Total** | **109** | All tests pass ✅ (2 skipped: ragas not installed) |
