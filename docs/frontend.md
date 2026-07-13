# Frontend

Frontend của Vietnam Tourism RAG được xây dựng bằng **React 19 + Vite 8 +
Tailwind CSS v4**.

## Cấu trúc

```
frontend/
├── src/
│   ├── api/
│   │   └── client.ts       # Client gọi API và xử lý SSE streaming
│   ├── components/
│   │   ├── ChatInput.tsx    # Input và nút gửi
│   │   ├── MessageBubble.tsx # Bong bóng tin nhắn user/assistant
│   │   └── CitationCard.tsx # Thẻ trích dẫn nguồn
│   ├── hooks/
│   │   └── useChat.ts      # Quản lý state chat
│   ├── types/
│   │   └── index.ts        # TypeScript types
│   ├── App.tsx             # Layout chính
│   ├── main.tsx            # Entry point
│   └── index.css           # Tailwind + custom styles
├── index.html
├── vite.config.ts
└── package.json
```

## Components

### ChatInput
- Textarea tự động mở rộng theo nội dung.
- Nút gửi và trạng thái loading.

### MessageBubble
- User: bong bóng xám, căn phải.
- Assistant: văn bản thuần, căn trái.
- Hiển thị thông báo progress (rewrite, retrieval, context, generation).
- Indicator động khi đang streaming.

### CitationCard
- Tag nguồn đánh số `[1]`, `[2]`, ...
- Hiển thị tiêu đề chunk khi hover.

## Kết nối API

Frontend chỉ gọi hai endpoint chính:

- `GET /api/health` — kiểm tra server.
- `POST /api/chat/stream` — streaming câu trả lời qua SSE.

Các SSE events từ backend:

```typescript
// Progress event
{ type: "progress", step: "rewrite", message: "Đang viết lại câu hỏi..." }

// Token event
{ type: "token", content: "Vịnh Hạ Long" }

// Done event
{ type: "done", answer: "...", sources: [...], intent: "factual" }

// Error event
{ type: "error", message: "Không đủ thông tin..." }
```

Trong development, Vite dev server proxy `/api` về backend tại
`http://localhost:8000`.

## Chạy local

```bash
cd frontend
npm install
npm run dev    # → http://localhost:5173
```

Backend cần chạy trước:

```bash
python -m rag_pipeline.api.app
```

## Build production

```bash
cd frontend
npm run build
```

Thư mục `frontend/dist/` được FastAPI serve tại root path (`/`). Khi deploy
qua Docker, multi-stage build trong `Dockerfile` sẽ tự động build frontend và
sao chép `dist/` vào image backend.
