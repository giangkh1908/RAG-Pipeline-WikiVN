# Chat Memory (No Auth)

Tính năng **memory cho chat** của RAG pipeline, hoạt động **không cần user đăng nhập**. Mỗi phiên chat được xác định bằng một `session_id` ẩn danh do frontend sinh và lưu `sessionStorage` — đóng tab hoặc reload là mất phiên.

Tài liệu này bổ sung cho [`docs/memory-plan.md`](memory-plan.md) (bản plan/spec gốc) bằng chi tiết implementation, hợp đồng API và ví dụ sử dụng thực tế.

---

## 1. Tổng quan kiến trúc

```
┌──────────┐  session_id (sessionStorage)   ┌──────────────────┐
│ Frontend │ ────────────────────────────▶ │  FastAPI /chat   │
│ (React)  │ ◀──── SSE stream events ────── │   (RAGPipeline)  │
└──────────┘                                  └────────┬─────────┘
                                                       │ persist
                                                       ▼
                          ┌──────────────────────────────────────┐
                          │  SQLite  (data/rag_storage.db)       │
                          │   ├─ chat_sessions                    │
                          │   └─ chat_turns (raw + summary)       │
                          └──────────────────────────────────────┘
                                                       ▲
                                                       │ summarize
                          ┌────────────────────────────┴─────────┐
                          │  MemoryCompactor (LLM via OpenRouter) │
                          └──────────────────────────────────────┘
```

Các thành phần chính:

| Module | Vai trò |
|---|---|
| `storage/conversation.py` | `ConversationStore` — DAO cho `chat_sessions` + `chat_turns` |
| `generation/memory.py` | `ConversationMemory` — assemble LLM message list + check threshold |
| `generation/compactor.py` | `MemoryCompactor` — gọi LLM tóm tắt, mutex per session, cache |
| `generation/rag_pipeline.py` | `RAGPipeline.answer_stream` — tích hợp memory + compact |
| `api/routes/chat.py` | Endpoints `/chat`, `/chat/stream`, `/session/{id}`, `/session/gc` |
| `api/schemas.py` | `ChatRequest` (max 500 chars + session_id), `ChatResponse`, `StreamDone` |
| Frontend `hooks/useChat.ts` | Sinh `session_id` ẩn danh, gắn vào request, nút "Mới" → reset |

---

## 2. Sơ đồ luồng một turn

```
User nhập question
        │
        ▼
[FE] useChat.sendMessage(question)
        │  session_id (sessionStorage)
        ▼
[BE] POST /api/chat
        │  1. Validate question (1-500 chars) + session_id (UUID-like)
        │  2. Resolve session_id: nếu thiếu → tự sinh, upsert_session
        │  3. Insert user turn (answer=NULL, turn_no=next)
        ▼
RAGPipeline.answer_stream(query, session_id)
        │
        │  4. preprocess query (rewrite, intent)
        │  5. retrieve → top-K chunks
        │  6. build context
        │
        ├── memory.enabled + session_id?
        │     │
        │     ▼
        │   ConversationMemory.build_history
        │     │  a. load_completed_turns
        │     │  b. raw_tokens = Σ est_tokens
        │     │  c. if raw_tokens >= threshold → compactor.compact
        │     │  d. assemble [system + RAG] + [summary?] + [N turn raw] + [question]
        │     │
        │     ▼
        │   LLMAnswerGenerator.generate_stream_messages(messages)
        │     │
        │     ▼
        │   ← stream tokens
        │
        │  7. Update turn.answer, intent, tokens_hint
        │  8. session.token_total += tokens_hint
        ▼
[FE] render tokens → khi 'done' cập nhật UI + lưu session_id mới (nếu server đổi)
```

---

## 3. Công thức token budget

Các con số chốt trong [`MemoryConfig`](../src/rag_pipeline/config.py):

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `max_input_chars` | 500 | Hard cap kích thước user question |
| `max_output_tokens` | 800 | `LLMAnswerGenerator.max_tokens` |
| `keep_raw_turns` | 3 | Số turn gần nhất giữ raw khi compact |
| `char_per_token` | 3 | Heuristic ước lượng token tiếng Việt |
| `summary_max_tokens` | 256 | Token budget cho câu summary |
| `session_ttl_hours` | 24 | Lazy GC sessions không hoạt động |
| `summary_model_name` | `openai/gpt-4o-mini` | Model dùng để tóm tắt |

```python
# Runtime threshold:
max_input_tokens   = ceil(max_input_chars / char_per_token)  # ≈ 167
single_turn_max    = max_input_tokens + max_output_tokens     # ≈ 967
memory_budget      = keep_raw_turns * single_turn_max         # ≈ 2900
compact_threshold  = 0.7 * memory_budget                      # ≈ 2030
```

Ngưỡng 70% đảm bảo: sau khi compact (giữ 3 turn raw), tổng token LLM call worst case luôn dưới 6K — rất an toàn với 128K context của `gpt-4o-mini`.

---

## 4. Compact — cách hoạt động

### 4.1. Khi nào trigger

```python
raw_tokens = sum(est_tokens(turn.question + turn.answer) for turn in all_turns)
if raw_tokens >= compact_threshold:
    compactor.compact(session_id)
```

### 4.2. Thuật toán compactor

1. **Mutex** — `acquire_compact_lock(session_id)` set `compacting=1`. Nếu request khác đang compact → trả `None`, caller dùng fallback (chỉ truncate).
2. **Re-read state** — sau khi có lock, đọc lại `all_turns` + `load_latest_summary_with_turn` (tránh stale state).
3. **Tính cutoff**:
   - Nếu `len(turns) <= keep_raw_turns` → return existing summary, không compact.
   - Ngược lại, `cutoff_turn_no = all_turns[-(keep + 1)].turn_no` (turn cuối trước nhóm giữ lại).
4. **Xác định turn cần summarize**:
   - Nếu đã có summary cũ ở turn `s`: `turns with s < turn_no <= cutoff`.
   - Nếu chưa có: `all_turns[:-keep]`.
5. **Build prompt** — gộp `Tóm tắt trước: ...` (nếu có) + từng cặp `User: ...\nAssistant: ...`.
6. **LLM call** — non-streaming, `max_tokens=256`, `temperature=0.2`, retry `summary_max_retries=3` với exponential backoff.
7. **Cache** — `UPDATE chat_turns SET summary=? WHERE session_id=? AND turn_no=?` (lưu trên turn có `turn_no` cao nhất trong range).
8. **Release lock** — kể cả khi fail.
9. **FailGrace** — nếu LLM fail, trả về `old_summary` (hoặc `None`) để caller dùng fallback.

### 4.3. Ví dụ timeline

Giả sử `keep_raw_turns=3`. Một session điển hình:

| Bước | Turn mới | Số turn | Raw tokens | Hành động |
|---|---|---|---|---|
| 1 | T1 | 1 | 200 | (turn đầu) |
| 2 | T2 | 2 | 400 | chưa compact |
| 3 | T3 | 3 | 600 | chưa compact |
| 4 | T4 | 4 | 800 | chưa compact |
| 5 | T5 | 5 | 1000 | chưa compact |
| 6 | T6 | 6 | 1200 | chưa compact (still < 2030) |
| 7 | T7 | 7 | 1400 | chưa compact |
| 8 | T8 | 8 | 1600 | chưa compact |
| 9 | T9 | 9 | 1800 | chưa compact |
| 10 | T10 | 10 | **2000** | chưa compact (< 2030) |
| 11 | T11 | 11 | **2200** | **compact** → summarize T1-T8, giữ T9-T11 raw |
| 12 | T12 | 12 | 1100 (raw) + summary | gửi `[summary] + [T10,T11,T12 raw]` |
| 13 | T13 | 13 | 1300 (raw) | chưa compact |
| 14 | T14 | 14 | 1500 (raw) | chưa compact |
| ... | | | | |

Khi raw history (cộng dồn) lại chạm 2030, compactor chỉ cần summarize các turn mới (T12 → sẽ gộp vào summary cũ) chứ không re-summarize toàn bộ.

---

## 5. Hợp đồng API

### 5.1. POST `/api/chat`

```http
POST /api/chat
Content-Type: application/json

{
  "question": "Vịnh Hạ Long ở đâu?",   // 1-500 chars
  "session_id": "abc123def456"          // optional, 8-64 chars [A-Za-z0-9_-]
}
```

**Response 200:**

```json
{
  "answer": "Vịnh Hạ Long nằm ở tỉnh Quảng Ninh.",
  "sources": [],
  "intent": "factual",
  "latency_ms": 1234.5,
  "session_id": "abc123def456",
  "turn_no": 3,
  "memory_used": true
}
```

**Lỗi:**

| Status | Ý nghĩa |
|---|---|
| 422 | `question` trống/quá dài, hoặc `session_id` sai regex |

### 5.2. POST `/api/chat/stream`

```http
POST /api/chat/stream
Content-Type: application/json
Accept: text/event-stream

{ "question": "...", "session_id": "..." }
```

**Response 200** (`text/event-stream`):

```
data: {"type": "progress", "step": "rewrite", "message": "Đang viết lại câu hỏi..."}

data: {"type": "progress", "step": "retrieval", "message": "Đang tìm kiếm thông tin..."}

data: {"type": "token", "content": "Vịnh "}

data: {"type": "token", "content": "Hạ Long"}

data: {"type": "done", "answer": "Vịnh Hạ Long nằm ở...", "sources": [], "intent": "factual", "session_id": "abc123def456", "turn_no": 3, "memory_used": true}
```

### 5.3. DELETE `/api/session/{session_id}`

Xóa cứng session + tất cả turns.

```http
DELETE /api/session/abc123def456
```

```json
{ "deleted": true, "deleted_turns": 5 }
```

### 5.4. POST `/api/session/gc`

Xóa sessions không hoạt động quá `session_ttl_hours` (24h mặc định). Chỉ xóa session không có pending turn (`answer IS NULL`).

```http
POST /api/session/gc
```

```json
{ "deleted": 12 }
```

---

## 6. Frontend tích hợp

### 6.1. Quản lý session_id

```ts
// src/api/client.ts
const SESSION_STORAGE_KEY = 'rag.session.id';

export function getOrCreateSessionId(): string {
  const existing = sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (existing) return existing;
  const fresh = crypto.randomUUID().replace(/-/g, '');
  sessionStorage.setItem(SESSION_STORAGE_KEY, fresh);
  return fresh;
}

export function clearSessionId(): string {
  const fresh = crypto.randomUUID().replace(/-/g, '');
  sessionStorage.setItem(SESSION_STORAGE_KEY, fresh);
  return fresh;
}
```

- Lưu ở `sessionStorage` (không phải `localStorage`) → reload tab = mất session, đúng yêu cầu ephemeral.
- Mỗi request `chatStream(question, sessionId, onEvent)` gắn `session_id` vào body.
- Khi nhận `done.session_id` khác với client có → cập nhật ref (server có thể tự sinh nếu client thiếu).

### 6.2. UX "Mới" (reset session)

```ts
const resetSession = useCallback(() => {
  sessionIdRef.current = clearSessionId();
  setMessages([]);
}, []);
```

Nút "Mới" trong header gọi `resetSession()` → gen session_id mới + clear messages. Phiên cũ vẫn còn trong DB nhưng sẽ bị GC sau 24h.

### 6.3. Input giới hạn 500 ký tự

```tsx
<textarea
  value={value}
  onChange={e => setValue(e.target.value.slice(0, 500))}
  maxLength={500}
  ...
/>
{value.length > 450 && (
  <span className={remaining < 20 ? 'text-red-500' : ''}>
    {remaining}
  </span>
)}
```

Counter chỉ hiện khi gần đầy (≥ 450 chars) để tránh noise.

---

## 7. Cấu hình & tuỳ chỉnh

Để thay đổi ngưỡng compact / số turn giữ raw / TTL, sửa dataclass [`MemoryConfig`](../src/rag_pipeline/config.py):

```python
from dataclasses import replace
from rag_pipeline.config import RAGConfig

config = RAGConfig()
config.memory.keep_raw_turns = 5         # giữ 5 turn raw thay vì 3
config.memory.max_input_chars = 800      # cho phép câu hỏi dài hơn
config.memory.summary_max_tokens = 512   # summary dài hơn
config.memory.session_ttl_hours = 72    # giữ session 3 ngày
```

Threshold sẽ tự động re-scale theo `max_input_chars` + `max_output_tokens`:

```
threshold = 0.7 * keep_raw_turns * (ceil(max_input_chars/3) + max_output_tokens)
```

---

## 8. Cách test

### 8.1. Unit tests (pytest)

```bash
# Tất cả tests
pytest

# Chỉ memory layer
pytest tests/test_memory.py tests/test_compactor.py -v
```

### 8.2. Smoke test (in-process FastAPI)

```bash
python scripts/smoke_memory.py
```

Output mẫu:

```
=== 1. Health check ===
  health = {'status': 'ok', 'qdrant': 'connected', 'version': '0.3.0'}
  ✓ health ok

=== 2. Chat without session_id (server auto-creates one) ===
  session_id = 9c1f1f3c503f42efaa825075cdcf3d74
  ...
  ✓ chat works, server mints session_id on the fly

=== 3. Chat with same session_id (memory continuity) ===
  turn_no    = 2 (incremented)
  ✓ session persists turn counter + memory flag

...

ALL SMOKE TESTS PASSED
```

Smoke test dùng `TestClient` + fake pipeline → không cần Qdrant/LLM thật, chạy được trong CI.

### 8.3. Manual test với server thật

```bash
# Start server (cần Qdrant + OPENROUTER_API_KEY)
python -m rag_pipeline.api.app

# Terminal khác:
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Vịnh Hạ Long ở đâu?"}'
# → trả về session_id

# Dùng session_id đó cho câu tiếp theo:
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Đi mùa nào đẹp?", "session_id": "<id trên>"}'
# → turn_no = 2, memory_used = true
```

---

## 9. Pitfalls đã biết

1. **Thread-safety SQLite** — `SQLiteStorage` dùng `check_same_thread=False` + WAL. Compactor chạy trong thread pool của FastAPI; an toàn cho concurrent read + serialize write.
2. **Race 2 request same session** — `UNIQUE(session_id, turn_no)` chặn duplicate. Hiếm gặp vì FE dùng `sessionStorage` (per-tab).
3. **Compactor LLM fail** — fallback graceful: trả về summary cũ (hoặc `None`); caller dùng truncation. Không bao giờ crash chat.
4. **Token estimate tiếng Việt** — `// 3` overestimate ~10-15% → an toàn về phí tràn. Có thể upgrade sang `tiktoken` nếu cần chính xác.
5. **Compact synchronous** — chặn request hiện tại 1-3s khi trigger. Nếu latency là vấn đề, có thể chuyển sang background task (out of scope MVP).
6. **Không có hydrate FE sau reload** — vì sessionStorage mất sau reload. Có thể thêm bằng cách lưu messages vào sessionStorage (out of scope MVP).
7. **`answer IS NULL` (pending turn) sẽ block GC** — bảo vệ turn đang stream khỏi bị xóa giữa chừng.

---

## 10. Roadmap

Các cải tiến tiềm năng (ngoài MVP):

- [ ] **Hydrate history FE** — lưu `messages` vào sessionStorage để restore sau reload.
- [ ] **Async compact** — chạy compactor ở background task thay vì block request.
- [ ] **Virtual scroll** — khi `messages.length` > ~100, dùng `@tanstack/react-virtual` để tránh lag DOM.
- [ ] **Token count chính xác** — tích hợp `tiktoken` cho OpenAI-compatible models.
- [ ] **Persistence channel cho compact cache** — khi summary lớn (>256 tokens), lưu file JSON riêng thay vì column `summary`.
- [ ] **User auth** — thay session ẩn danh bằng user_id; memory + summary theo user, persistent lâu dài.
