# Conversation Memory (v1)

Hệ thống hỗ trợ multi-turn chat — LLM hiểu context từ lịch sử hội thoại trước đó.

> **v1**: Client-side history, LLM trực tiếp resolve context. V2 sẽ dùng Agent + server-side memory.

## Vấn đề

RAG pipeline gốc xử lý mỗi câu hỏi là independent:

```
User: "Thủ đô Việt Nam ở đâu?"
Bot:  "Thủ đô Việt Nam là Hà Nội."

User: "Dân số bao nhiêu?"
Bot:  "Dân số nước nào?"  ← Không biết "nước nào" = Việt Nam
```

## Giải pháp

Gửi conversation history từ frontend → backend cùng với câu hỏi mới. LLM thấy toàn bộ context:

```
System: Bạn là trợ lý AI...
User:   "Thủ đô Việt Nam ở đâu?"
Asst:   "Thủ đô Việt Nam là Hà Nội."
User:   "Dân số bao nhiêu?"  + [passages về Việt Nam]
→ LLM biết "Dân số" = dân số Việt Nam
```

## Kiến trúc

```
Frontend                           Backend
┌─────────────────────┐           ┌──────────────────────────┐
│ messages[]           │           │                          │
│  [0] user: "..."     │  POST    │ ChatRequest              │
│  [1] asst: "..."     │ ────────▶│  question: "..."         │
│  [2] user: "..."     │  JSON    │  history: [{role, content}]│
│  [3] asst: "..."     │           │                          │
│                      │           │ PromptBuilder             │
│ useChat hook         │           │  → trim_history()        │
│  extractHistory()    │           │  → inject vào messages   │
│  max 5 turns         │           │                          │
└─────────────────────┘           │ AnswerGenerator           │
                                  │  → LLM với full context  │
                                  └──────────────────────────┘
```

## Components

### 1. Frontend — History Extraction

File: `frontend/src/hooks/useChat.ts`

```typescript
const MAX_HISTORY_TURNS = 5;

function extractHistory(messages: Message[]): ChatHistoryEntry[] {
  const completed = messages.filter(m => !m.isStreaming);
  const history: ChatHistoryEntry[] = [];
  for (const m of completed) {
    if (m.role === 'user' || (m.role === 'assistant' && m.content)) {
      history.push({ role: m.role, content: m.content });
    }
  }
  return history.slice(-(MAX_HISTORY_TURNS * 2));
}
```

- Lọc bỏ messages đang streaming
- Chuyển `Message[]` → `ChatHistoryEntry[]` (role + content)
- Giới hạn 5 turns gần nhất (10 messages)

### 2. Frontend — API Client

File: `frontend/src/api/client.ts`

```typescript
export async function chatStream(
  question: string,
  onEvent: (event: StreamEvent) => void,
  history: ChatHistoryEntry[] = [],
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history }),
  });
  // ...stream handling
}
```

- Đổi từ GET → POST (history có thể dài, không fit vào URL)
- Gửi `{question, history}` trong body

### 3. Backend — Schema

File: `src/rag_pipeline/api/schemas.py`

```python
class ChatHistoryEntry(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    question: str
    history: list[ChatHistoryEntry] = []  # optional, default rỗng
    use_reranker: bool = False
    use_llm: bool = True
```

### 4. Backend — Context Window Management

File: `src/rag_pipeline/generation/prompt_builder.py`

Hệ thống quản lý history bằng 2 chiến lược: **Summarization** (khi history dài) + **Truncation** (khi vượt token budget).

#### Strategy Overview

```
History <= 6 turns  →  Gửi full history (giống ChatGPT/Claude)
History > 6 turns   →  Summarize older + giữ 2 turns gần nhất
Vẫn vượt budget    →  Truncate từ cũ nhất
```

So sánh với ChatGPT/Claude:

| Approach | ChatGPT | Claude | RAG v1 |
|----------|---------|--------|--------|
| Full history | Có (128K window) | Có (200K window) | Có (16K window) |
| Summarization | Có (khi gần đầy) | Không | Có (khi > 6 turns) |
| Truncation | Có (fallback) | Có | Có (fallback) |
| Token budget | Tự động | Tự động | 16K (configurable) |

#### Token Budget

```
max_context_tokens (16,000)
├── Reserved: system message    (~600 tokens)
├── Reserved: passages          (~4,000 tokens)
├── Reserved: question + format (~300 tokens)
├── Reserved: response          (~2,000 tokens)
└── History budget              (~9,100 tokens)
```

#### Summarization (khi history dài)

```python
_SUMMARIZE_THRESHOLD = 6   # Summarize khi > 6 turns
_KEEP_RECENT_TURNS = 2     # Giữ 2 turns gần nhất

# Flow:
# [turn1, turn2, turn3, turn4, turn5, turn6, turn7, turn8]
# → [summary(turn1-6), turn7, turn8]
```

LLM tóm tắt older turns thành 2-3 câu, giữ lại thông tin quan trọng (chủ đề, entities, facts).

```
Input (6 turns):
- User: Thủ đô Việt Nam ở đâu?
- Asst: Thủ đô Việt Nam là Hà Nội.
- User: Dân số bao nhiêu?
- Asst: Dân số Hà Nội là 8,053,663.
- User: Có thông tin năm 2025 ko?
- Asst: Không tìm thấy thông tin năm 2025.

Output (summary):
"[Tóm tắt 6 lượt hội thoại trước: Người dùng hỏi về thủ đô Việt Nam (Hà Nội) và dân số Hà Nội (8,053,663 người). Không có thông tin dân số năm 2025.]"
```

#### Truncation (fallback)

Nếu summarize không khả dụng hoặc vẫn vượt budget → truncate từ cũ nhất:

```python
def _trim_history(self, history, system_msg, user_msg):
    reserved = estimate(system_msg) + estimate(user_msg) + RESERVED_RESPONSE
    history_budget = max_context_tokens - reserved

    # Walk từ mới → cũ, accumulate cho đến khi vượt budget
    total = 0
    keep_from = len(history)
    for i in range(len(history) - 1, -1, -1):
        turn_tokens = estimate(history[i]["content"])
        if total + turn_tokens > history_budget:
            break
        total += turn_tokens
        keep_from = i

    return history[keep_from:]
```

#### Token Estimation

```python
_CHARS_PER_TOKEN = 4  # ~4 chars/token cho tiếng Việt + English

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)
```

#### Configuration

```python
@dataclass
class GenerationConfig:
    max_context_tokens: int = 16_000  # Context window budget
```

PromptBuilder nhận `llm_client` để gọi LLM cho summarization:

```python
prompt_builder = PromptBuilder(gen_config, llm_client=llm_client)
```

Ước tính rough nhưng đủ chính xác cho truncation. Sai lệch ±20% acceptable vì budget có margin.

### 5. Backend — Prompt Injection

History được inject giữa system message và user message:

```
messages = [
    {"role": "system", "content": "Bạn là trợ lý AI..."},
    {"role": "user", "content": "Thủ đô Việt Nam ở đâu?"},
    {"role": "assistant", "content": "Thủ đô Việt Nam là Hà Nội."},
    {"role": "user", "content": "NGỮ CẢNH: ...\nCÂU HỎI: Dân số bao nhiêu?"},
]
```

## Configuration

File: `src/rag_pipeline/config.py`

```python
@dataclass
class GenerationConfig:
    max_answer_tokens: int = 1024
    temperature: float = 0.1
    max_context_tokens: int = 16_000  # ← context window budget
```

### Tùy chỉnh theo model

| Model | Context Window | `max_context_tokens` |
|-------|---------------|---------------------|
| DeepSeek V4 Flash | 128K | `16_000` (default, đủ cho 5 turns) |
| GPT-4o | 128K | `32_000` (nếu muốn nhiều history hơn) |
| Llama 3.2 3B | 128K | `16_000` |
| Small models (< 8K) | 8K | `4_000` (ít history) |

## API

### POST /api/chat (non-streaming)

```json
{
  "question": "Dân số bao nhiêu?",
  "history": [
    {"role": "user", "content": "Thủ đô Việt Nam ở đâu?"},
    {"role": "assistant", "content": "Thủ đô Việt Nam là Hà Nội."}
  ]
}
```

### POST /api/chat/stream (SSE streaming)

```json
{
  "question": "Dân số bao nhiêu?",
  "history": [
    {"role": "user", "content": "Thủ đô Việt Nam ở đâu?"},
    {"role": "assistant", "content": "Thủ đô Việt Nam là Hà Nội."}
  ]
}
```

Response: SSE stream (giống như trước, không thay đổi format).

## Behavior

### Khi có history

```
User: "Python là gì?"
Bot:  "Python là ngôn ngữ lập trình..."

User: "Ai tạo ra nó?"
→ History gửi kèm: [user: "Python là gì?", asst: "Python là..."]
→ LLM hiểu "nó" = Python
→ Trả lời: "Python được tạo bởi Guido van Rossum..."
```

### Khi history bị cắt

Nếu conversation dài, turns cũ bị drop:

```
[Đã lược bỏ 3 lượt hội thoại cũ để tiết kiệm ngữ cảnh]
User: Tiếp tục câu hỏi trước...
Asst: ...
User: Câu hỏi mới nhất
```

LLM biết context bị lược bỏ, sẽ trả lời dựa trên turns gần nhất.

### Khi không có history

Backward compatible — request không có `history` hoặc `history: []` hoạt động như cũ, không thay đổi behavior.

## Frontend Integration

### useChat hook

```typescript
const sendMessage = useCallback(async (question: string) => {
  // Extract history từ messages hiện tại
  const history = extractHistory(messages);

  // Gửi kèm history
  await chatStream(question, onEvent, history);
}, [isStreaming, messages]);
```

Dependency `[isStreaming, messages]` đảm bảo history luôn cập nhật.

### clearMessages

```typescript
const clearMessages = useCallback(() => {
  setMessages([]);
}, []);
```

Xóa messages = xóa history. Turns tiếp theo không có context trước đó.

## Query Rewriting with Context

History không chỉ dùng cho generation — còn dùng để **rewrite query trước khi retrieval**.

### Vấn đề

```
User: "Thủ đô Việt Nam ở đâu?"
Bot:  "Thủ đô Việt Nam là Hà Nội."

User: "Dân số chúng là bao nhiêu?"
→ BM25 search: "dân số chúng bao nhiêu"  ← không có "Việt Nam"
→ Retrieval sai: trả về kết quả về "Tam Quốc", "Chư Thành"
```

### Giải pháp

Khi có history, query rewriter dùng LLM để resolve đại từ:

```
History: "Thủ đô Việt Nam ở đâu?" → "Hà Nội"
Question: "Dân số chúng là bao nhiêu?"
→ LLM rewrite: "Dân số Việt Nam là bao nhiêu?"
→ BM25 search: "dân số việt nam"  ← đúng!
```

### Flow

```
Question + History
    │
    ▼
QueryRewriter.rewrite(query, history)
    │
    ├── Nếu có history → LLM rewrite với context
    │   Prompt: "Ngữ cảnh: ...\nCâu hỏi: ..."
    │   → Resolve đại từ, expand abbreviations
    │
    └── Nếu không có history → Fast path (skip rewrite)
        → Tiết kiệm ~4s
    │
    ▼
ProcessedQuery (rewrite_query, bm25_query)
    │
    ▼
Retrieval (dense + BM25 với query đã resolve)
```

### Implementation

File: `src/rag_pipeline/query/rewriter.py`

```python
REWRITE_PROMPT = """...
{history_section}Cho câu hỏi sau:
"{query}"

1. "normalized_query": ... Thay đại từ mập mờ bằng danh từ cụ thể dựa trên ngữ cảnh.
2. "rewrite_query": ... Thay đại từ bằng danh từ cụ thể.
3. "bm25_query": ... Nếu có đại từ, thay bằng danh từ cụ thể.
..."""

def rewrite(self, query: str, history: list[dict] | None = None) -> RewriteResult:
    history_section = self._format_history(history)
    prompt = REWRITE_PROMPT.format(query=query, history_section=history_section)
    ...
```

### Behavior

| Scenario | Rewrite | Thời gian |
|----------|---------|-----------|
| Không có history | Skip (fast path) | ~0ms |
| Có history, câu đơn giản | LLM rewrite | ~4s |
| Có history, có đại từ | LLM resolve + rewrite | ~4s |

**Trade-off**: Chậm hơn ~4s khi có history, nhưng retrieval chính xác hơn nhiều.

## Limitations

1. **Client-side only** — History nằm trong React state, reload page = mất history
2. **Token estimation** — ~4 chars/token là ước tính, có thể sai ±20%
3. **No server-side sessions** — Không lưu conversation trên server
4. **LLM rewrite latency** — Thêm ~4s khi có history (đổi lấy accuracy)

## Future Improvements

- **Server-side sessions**: Lưu conversation theo `conversation_id`, frontend chỉ cần gửi ID
- **Persistent memory**: Lưu conversations vào database cho users đã đăng nhập
- **Smarter fast path**: Detect đại từ bằng regex, chỉ gọi LLM rewrite khi cần
