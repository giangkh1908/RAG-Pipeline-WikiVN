# Frontend Architecture

React 19 + Vite 8 + Tailwind CSS v4

## Cấu trúc

```
frontend/
├── src/
│   ├── api/
│   │   └── client.ts       # SSE streaming client
│   ├── components/
│   │   ├── ChatInput.tsx    # Input + suggestions
│   │   ├── MessageBubble.tsx # User/Assistant bubbles
│   │   └── CitationCard.tsx # Source citations
│   ├── hooks/
│   │   └── useChat.ts      # Chat state management
│   ├── types.ts            # TypeScript types
│   ├── App.tsx             # Main layout
│   ├── main.tsx            # Entry point
│   └── index.css           # Tailwind + custom styles
├── index.html
├── vite.config.ts
└── package.json
```

## Components

### ChatInput
- Auto-expanding textarea
- Horizontal scroll suggestions (mobile)
- Arrow send button with loading spinner

### MessageBubble
- User: gray bubble, right-aligned (iMessage style)
- Assistant: plain text, left-aligned (ChatGPT style)
- Streaming indicator while loading

### CitationCard
- Numbered source tags [1] [2] [3]
- Wikipedia links
- Confidence scores

## SSE Streaming

Client kết nối đến `/api/chat/stream?question=...` và nhận events:

```typescript
// Token event
{ type: "token", content: "Hello" }

// Done event
{ type: "done", answer: "...", citations: [...], confidence: 0.8 }
```

## Responsive Design

| Breakpoint | Width | Layout |
|------------|-------|--------|
| Mobile | < 640px | Horizontal scroll suggestions |
| Tablet | 640-1024px | Wrap suggestions |
| Desktop | > 1024px | Max-width 768px centered |

## Build

```bash
# Development
npm run dev    # → http://localhost:5173

# Production
npm run build  # → dist/
```

Production build được serve bởi FastAPI (không cần Nginx).
