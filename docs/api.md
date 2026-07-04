# API Reference — FastAPI Backend (v1)

Backend HTTP + SSE cho RAG Pipeline. React frontend kết nối qua API này.

> **v1**: API gọi LLM trực tiếp, single-turn hoặc multi-turn với client-side history.

## Quick Start

```bash
# Start server
python -m rag_pipeline.api.app

# Hoặc dùng uvicorn trực tiếp
uvicorn rag_pipeline.api.app:app --host 0.0.0.0 --port 8000 --reload

# Swagger UI
open http://localhost:8000/docs
```

## Endpoints

### Health Check

```http
GET /api/health
```

Response:
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "langsmith_enabled": true,
  "components": {
    "qdrant": "ok",
    "langsmith": "enabled"
  }
}
```

### Chat (Non-streaming)

```http
POST /api/chat
Content-Type: application/json

{
  "question": "Python là gì?",
  "top_k": 5,
  "use_reranker": true,
  "use_llm": true
}
```

Response:
```json
{
  "answer": "Python là ngôn ngữ lập trình...",
  "citations": [
    {
      "doc_id": "123",
      "title": "Python (programming language)",
      "url": "https://vi.wikipedia.org/wiki/Python",
      "score": 0.92
    }
  ],
  "confidence": 0.85,
  "passages_used": 5,
  "latency_ms": 3245.2
}
```

### Chat (SSE Streaming)

```http
GET /api/chat/stream?question=Python+là+gì&top_k=5
```

SSE events:
```
data: {"type":"token","content":"Python"}

data: {"type":"token","content":" là"}

data: {"type":"token","content":" ngôn"}

data: {"type":"token","content":" ngữ"}

data: {"type":"done","citations":[{"doc_id":"123","title":"Python","url":"...","score":0.92}],"confidence":0.85}

```

### Run Evaluation

```http
POST /api/eval
Content-Type: application/json

{
  "max_questions": 10,
  "use_reranker": true,
  "use_llm": true
}
```

Response: RAGAS metrics JSON (faithfulness, answer_relevancy, context_precision, context_recall).

## SSE Streaming Format

### Token event
```json
{"type": "token", "content": "text chunk"}
```

### Done event
```json
{
  "type": "done",
  "citations": [...],
  "confidence": 0.85
}
```

### Error event
```json
{"type": "error", "message": "error description"}
```

## Frontend Integration (ReadableStream)

```typescript
const response = await fetch(`/api/chat/stream?question=${encodeURIComponent(question)}`);
const reader = response.body!.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const text = decoder.decode(value);
  const lines = text.split('\n');

  for (const line of lines) {
    if (!line.startsWith('data: ')) continue;
    const data = JSON.parse(line.slice(6));

    if (data.type === 'token') {
      // Append token to UI
      setAnswer(prev => prev + data.content);
    } else if (data.type === 'done') {
      // Show citations
      setCitations(data.citations);
    }
  }
}
```

## CORS

Allowed origins:
- `http://localhost:3000` — React dev server
- `http://localhost:5173` — Vite dev server
- `https://<domain>` — Production domain
- `https://www.<domain>` — Production domain (www)

## Architecture

```
React Frontend (port 5173)
    │
    │ HTTP / SSE
    ▼
FastAPI Backend (port 8000)
    │
    ├── /api/health → Qdrant + LangSmith check
    ├── /api/chat   → Pipeline.ask() → JSON
    ├── /api/chat/stream → Pipeline streaming → SSE
    └── /api/eval   → Pipeline.run_eval() → metrics
```

## Error Handling

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 422 | Validation error (wrong request body) |
| 500 | Internal server error |
| 503 | Service unavailable (Qdrant disconnected) |

## Development

```bash
# Install API dependencies
pip install -e ".[api]"

# Run with auto-reload
uvicorn rag_pipeline.api.app:app --reload

# Run tests
pytest tests/test_api.py -v
```
