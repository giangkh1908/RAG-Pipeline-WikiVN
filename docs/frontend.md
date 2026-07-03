# Frontend — React Chat Interface

React 19 chat UI kết nối FastAPI backend qua SSE streaming. Responsive cho mobile.

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
│   │   ├── ChatInput.tsx      # Input + suggestions + send button
│   │   ├── CitationCard.tsx   # Numbered source tag [1] [2]
│   │   └── MessageBubble.tsx  # User bubble / Assistant text
│   ├── hooks/
│   │   └── useChat.ts         # Chat state + SSE streaming
│   ├── types/
│   │   └── index.ts           # TypeScript interfaces
│   ├── App.tsx                # Root layout
│   ├── main.tsx               # Entry point
│   └── index.css              # Tailwind + scrollbar-hide
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

## Responsive Design

Mobile-first, hoạt động tốt trên mọi kích thước:

| Breakpoint | Changes |
|------------|---------|
| `< 640px` (mobile) | Suggestions horizontal scroll, input full-width, compact padding |
| `≥ 640px` (tablet+) | Suggestions wrap + center, larger padding |

**Mobile optimizations:**
- `h-[100dvh]` — full viewport, tránh browser bar
- `overscroll-behavior: none` — bỏ bounce iOS
- `-webkit-tap-highlight-color: transparent` — bỏ highlight xanh
- `active:bg-gray-100` — touch feedback
- `.scrollbar-hide` — ẩn scrollbar cho suggestions

## Components

### ChatInput
- Auto-expanding textarea (tối đa 160px)
- Nút gửi hình mũi tên (như ChatGPT)
- Spinner khi đang loading
- Gợi ý câu hỏi (click để gửi)
- Horizontal scroll trên mobile, wrap trên desktop

### MessageBubble
- **User**: bubble xám, căn phải (như iMessage)
- **Assistant**: text trần, căn trái (như ChatGPT/Claude)
- Streaming: 3 dấu chấm nhảy khi chờ, cursor nhấp nháy khi đang stream

### CitationCard
- Tag nhỏ có số [1] [2] [3]
- Tên bài viết Wikipedia
- Click mở link trong tab mới

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
