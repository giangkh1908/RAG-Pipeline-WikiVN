# RAG Pipeline — Vietnamese Wikipedia

Hỏi đáp dựa trên 1.1 triệu bài viết Wikipedia tiếng Việt, sử dụng RAG (Retrieval-Augmented Generation).

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Python |
| Frontend | React 19 + Vite 6 + Tailwind CSS v4 (responsive) |
| Vector Store | Qdrant |
| Embedding | OpenRouter (nvidia/llama-nemotron-embed-vl-1b-v2:free, 2048-dim) |
| LLM | OpenRouter (deepseek/deepseek-v4-flash) |
| Re-ranking | Cohere Rerank v3.5 |
| BM25 | rank-bm25 + pyvi |
| Tracing | LangSmith |
| Evaluation | RAGAS |

## Cấu trúc thư mục

```
RAG/
├── src/rag_pipeline/       # Python backend
│   ├── api/                # FastAPI server
│   │   ├── app.py          # App + CORS + static serving
│   │   ├── schemas.py      # Pydantic models
│   │   └── routes/         # API endpoints
│   ├── query/              # Query processing
│   ├── retrieval/          # Hybrid search
│   ├── generation/         # LLM answer generation
│   └── eval/               # RAGAS evaluation
├── frontend/               # React frontend
│   ├── src/
│   │   ├── api/client.ts   # SSE streaming client
│   │   ├── hooks/useChat.ts
│   │   └── components/     # UI components
│   └── dist/               # Build output (auto-served by FastAPI)
├── tests/                  # pytest tests
├── docs/                   # Documentation
└── docker-compose.yml      # Qdrant container
```

---

## Hướng dẫn chạy

### Bước 1: Clone & setup Python

```bash
git clone <repo-url>
cd RAG

# Tạo virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Cài dependencies
pip install -e ".[indexing,monitoring,eval,api]"
```

### Bước 2: Environment variables

Tạo file `.env` trong thư mục gốc:

```env
# Bắt buộc
OPENROUTER_API_KEY=sk-or-v1-xxx

# Qdrant (mặc định: localhost:6333)
QDRANT_URL=http://localhost:6333

# Tùy chọn: Cohere re-ranking
COHERE_API_KEY=xxx

# Tùy chọn: LangSmith tracing
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_xxx
LANGSMITH_PROJECT=rag-pipeline
LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com
```

### Bước 3: Start Qdrant

```bash
docker-compose up -d
```

### Bước 4: Ingest dữ liệu

```bash
# Test với sample nhỏ (nhanh, ~1 phút)
python -m rag_pipeline.main ingest --sample 0.001

# Full dataset (1.1M docs, mất vài giờ)
python -m rag_pipeline.main ingest
```

### Bước 5: Chạy ứng dụng

#### Cách A: Chạy riêng backend + frontend (development)

```bash
# Terminal 1 — Backend API
python -m rag_pipeline.api.app
# → http://localhost:8000

# Terminal 2 — Frontend dev server
cd frontend
npm install
npm run dev
# → http://localhost:5173 (tự proxy API sang backend)
```

#### Cách B: Chạy production (1 server)

```bash
# Build frontend trước
cd frontend
npm install
npm run build
cd ..

# Start API (tự serve frontend)
python -m rag_pipeline.api.app
# → http://localhost:8000 (cả API + frontend)
```

---

## Frontend

Chat UI responsive, hoạt động trên mobile và desktop.

**Features:**
- SSE streaming token-by-token
- Auto-expanding textarea
- Gợi ý câu hỏi (click để gửi)
- Citations hiển thị nguồn Wikipedia
- Responsive: mobile horizontal scroll, desktop wrap

**Truy cập từ điện thoại:**
```bash
# Tìm IP của máy
ipconfig  # Windows

# Truy cập từ điện thoại
http://<your-ip>:8000
```

---

## API Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| `GET` | `/api/health` | Kiểm tra trạng thái |
| `POST` | `/api/chat` | Hỏi đáp (JSON response) |
| `GET` | `/api/chat/stream?question=...` | Hỏi đáp (SSE streaming) |
| `POST` | `/api/eval` | Chạy đánh giá RAGAS |
| `GET` | `/docs` | Swagger UI |

### Ví dụ gọi API

```bash
# Non-streaming
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Python là gì?"}'

# SSE streaming (mặc định skip_rewrite=true, ~6-8s)
curl -N "http://localhost:8000/api/chat/stream?question=Python+la+gi"

# Full pipeline với query rewrite (~12-16s)
curl -N "http://localhost:8000/api/chat/stream?question=Python+la+gi&skip_rewrite=false"
```

### Streaming Modes

| Mode | Flag | TTFT | Khi nào dùng |
|------|------|------|---------------|
| Fast (default) | `skip_rewrite=true` | ~6-8s | Câu hỏi đơn giản |
| Full | `skip_rewrite=false` | ~12-16s | Câu phức tạp, cần query rewrite |

---

## CLI Commands

```bash
# Hỏi đáp (text output)
python -m rag_pipeline.main ask --question "Wikipedia là gì?" --text

# Hỏi đáp (streaming)
python -m rag_pipeline.main ask --question "Wikipedia là gì?" --text --stream

# Search documents
python -m rag_pipeline.main search --query "lịch sử Việt Nam"

# Đánh giá
python -m rag_pipeline.main eval --limit 10
```

---

## Tests

```bash
# Chạy tất cả tests
python -m pytest tests/ -v

# Chỉ chạy API tests
python -m pytest tests/test_api.py -v

# Bỏ qua eval tests (chậm)
python -m pytest tests/ -v -k "not eval"
```

---

## Documentation

| File | Nội dung |
|------|----------|
| [PLAN.md](PLAN.md) | Kế hoạch dự án & tiến độ |
| [docs/api.md](docs/api.md) | API reference |
| [docs/frontend.md](docs/frontend.md) | Frontend architecture |
| [docs/generation.md](docs/generation.md) | Generation pipeline |
| [docs/eval.md](docs/eval.md) | Evaluation metrics |

## Docker

Chạy toàn bộ hệ thống với Docker Compose (Qdrant + API + Frontend).

### Build & Run

```bash
# Tạo file .env từ mẫu
cp .env.example .env
# Sửa .env với API keys của bạn

# Build và start
docker-compose up --build -d

# Xem logs
docker-compose logs -f api

# Dừng
docker-compose down
```

Truy cập: `http://localhost:8000`

### Chỉ dùng Qdrant (phát triển local)

```bash
# Chỉ start Qdrant
docker-compose up -d qdrant

# Chạy backend + frontend như bình thường
python -m rag_pipeline.api.app
```

---

## License

MIT
