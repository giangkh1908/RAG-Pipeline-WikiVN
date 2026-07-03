# Frontend — React Chat Interface

React 19 chat UI kết nối FastAPI backend qua SSE streaming.

## Tech Stack

- React 19 + TypeScript
- Vite 6 (dev server + build)
- Tailwind CSS v4
- ReadableStream API (SSE consumption)

## Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── client.ts         # fetch + ReadableStream wrapper
│   ├── components/
│   │   ├── ChatInput.tsx      # Textarea + send button
│   │   ├── CitationCard.tsx   # Wikipedia source link
│   │   └── MessageBubble.tsx  # User/assistant message
│   ├── hooks/
│   │   └── useChat.ts         # Chat state + SSE streaming logic
│   ├── types/
│   │   └── index.ts           # TypeScript interfaces
│   ├── App.tsx                # Root component
│   ├── main.tsx               # Entry point
│   └── index.css              # Tailwind import
├── package.json
├── vite.config.ts             # Tailwind plugin + API proxy
└── tsconfig.json
```

## Development

```bash
cd frontend
npm install
npm run dev        # → http://localhost:5173
```

Vite proxy tự chuyển `/api/*` → `http://localhost:8000` (không cần CORS).

## Build + Serve

```bash
# Build frontend
cd frontend && npm run build

# Start FastAPI (tự serve frontend/dist nếu tồn tại)
python -m rag_pipeline.api.app
# → http://localhost:8000
```

## SSE Streaming Flow

```
User types question
    ↓
useChat.sendMessage()
    ↓
fetch /api/chat/stream?question=...
    ↓
ReadableStream reads SSE events
    ↓
onEvent({type:'token', content:'...'}) → append to message
onEvent({type:'done', citations:[...]}) → show sources
```

## Key Files

### `api/client.ts`
- `chat(question)` — POST, returns full response
- `chatStream(question, onEvent)` — GET SSE, calls onEvent per stream event
- `healthCheck()` — GET /api/health

### `hooks/useChat.ts`
- `messages` — Message[] state
- `isStreaming` — boolean
- `sendMessage(question)` — triggers SSE stream
- `clearMessages()` — reset chat

### `types/index.ts`
- `Message` — {id, role, content, citations?, confidence?, isStreaming?}
- `StreamEvent` — StreamToken | StreamDone | StreamError
- `Citation` — {doc_id, title, url, score}
