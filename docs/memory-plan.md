# Plan: Chat Memory (No Auth, Ephemeral Frontend)

## 1. Tong quan

He thong them **memory cho chat** ma khong yeu cau authentication. Dung **session an danh** (frontend sinh UUID, luu `sessionStorage` — reload/thoat tab la mat session). Backend persist vao SQLite de ho tro compact + TTL, khong luu RAM de tranh mat khi restart backend.

**Nguyen tac:**
- Memory **opt-in**: request khong co `session_id` → flow stateless cu, backward compatible.
- Memory tach bai khoi RAG corpus (Qdrant) — chi luu trong SQLite.
- Compact dua tren **token budget**, khong dung `max_turns` co dinh.
- Bo hien thi citation (khong co link xac thuc, chi co text).
- Khong che input (500 ky tu) + output (800 tokens).

---

## 2. Tham so chot

| Tham so | Gia tri | Ghi chu |
|---|---|---|
| Max input | **500 ky tu** | Validate o `ChatRequest` |
| Max output | **800 tokens** | `GenerationConfig.max_tokens` (doi tu 1024) |
| Token estimator | `len(text) // 3` | Heuristic cho tieng Viet (tinh rong) |
| Keep raw turns khi compact | **3** | So turn gan nhat giu nguyen |
| Compact threshold | **2100 tokens** | 70% x memory_budget |
| Memory budget | **3000 tokens** | 3 x (max_input_tokens + max_output_tokens) = 3 x (167 + 800) |
| Summary max tokens | **256** | |
| Summary temperature | **0.2** | |
| Session FE storage | `sessionStorage` | Reload/thoat tab = mat session |
| Session BE TTL | **24h** | Lazy GC |
| Citations | **Khong hien thi** | Bo UI CitationCard |

### Cong thuc tinh budget (runtime, tu scale)

```
est_tokens(text) = len(text) // 3

max_input_tokens   = est_tokens(MAX_INPUT_CHARS)    # 500 // 3 ≈ 167
max_output_tokens  = MAX_OUTPUT_TOKENS              # 800
single_turn_max    = max_input_tokens + max_output_tokens  # ≈ 967
memory_budget      = KEEP_RAW_TURNS x single_turn_max      # 3 x 967 ≈ 2900
compact_threshold  = 0.7 x memory_budget                     # ≈ 2030 → lam tron 2100
```

→ Neu sau nay doi max_input/max_output, threshold **tu dong scale theo**.

### Kiem tra tong token LLM call worst case (sau compact)

```
system_prompt(~100) + RAG_context(~1500) + summary(~256)
+ 3_turn_raw(~3000) + current_question(~167) + output(800)
= ~5823 tokens
```

→ Xa cap 128K cua gpt-4o-mini. An toan.

---

## 3. DB Schema (SQLite, cung file `data/rag_storage.db`)

Them 2 bang trong `_ensure_schema()` cua `storage/sqlite.py`. Dung `CREATE TABLE IF NOT EXISTS` → backward compatible.

### Bang `chat_sessions`

| Cot | Kieu | Mo ta |
|---|---|---|
| `session_id` | TEXT PK | UUID do FE sinh |
| `created_at` | TEXT | ISO8601 |
| `last_active_at` | TEXT | ISO8601, update moi request |
| `token_total` | INTEGER DEFAULT 0 | Tong token raw history (quick check compact) |
| `compacting` | INTEGER DEFAULT 0 | Mutex flag (0=idle, 1=dang compact) |

### Bang `chat_turns`

| Cot | Kieu | Mo ta |
|---|---|---|
| `id` | TEXT PK | UUID |
| `session_id` | TEXT NOT NULL | FK logic → chat_sessions |
| `turn_no` | INTEGER NOT NULL | 1, 2, 3… tang theo session |
| `question` | TEXT NOT NULL | Max 500 chars (check API) |
| `answer` | TEXT | NULL khi chua xong, update sau LLM |
| `intent` | TEXT | Tu retrieval preprocessor |
| `tokens_hint` | INTEGER | Estimation cua (question + answer) |
| `summary` | TEXT | NULL default; update khi compact |
| `created_at` | TEXT NOT NULL | ISO8601 |
| INDEX + UNIQUE | `(session_id, turn_no)` | 1 turn / session khong trung |

### Pragma

- `PRAGMA journal_mode=WAL` → concurrent read + serialize write.

---

## 4. Nghiep vu moi turn (flow chi tiet)

### Buoc 1 — Validate (API layer)

- `ChatRequest.question`: `min_length=1`, `max_length=500` (sua tu 1000 xuong).
- Neu co `session_id`: validate regex `^[A-Za-z0-9_-]{8,64}$`.
- Tra `422` neu vi pham.

### Buoc 2 — Session upsert

- `INSERT INTO chat_sessions … ON CONFLICT(session_id) DO UPDATE SET last_active_at=now`.
- Neu khong co `session_id` → BE sinh UUID, tra ve response field `session_id` cho FE luu.

### Buoc 3 — Insert user turn

- `turn_no = COALESCE(MAX(turn_no), 0) + 1` trong session.
- `answer = NULL` tam (update sau khi LLM xong).
- Dung transaction cung connection tranh race.

> **Ly do insert truoc**: neu LLM fail van co audit; FE thay cau hoi da gui.

### Buoc 4 — Load memory

```
summary = SELECT summary FROM chat_turns
          WHERE session_id=? AND summary IS NOT NULL
          ORDER BY turn_no ASC LIMIT 1

raw_turns = SELECT * FROM chat_turns
            WHERE session_id=? AND answer IS NOT NULL
            ORDER BY turn_no ASC
```

### Buoc 5 — Tinh token + quyet dinh compact

```
raw_tokens = Σ est(question + answer) cho tat ca raw_turns

IF raw_tokens < compact_threshold:
    → gui TAT CA raw_turns vao LLM (khong compact)
    → co the la 4, 5, 7+ turn tuy do dai thuc te

IF raw_tokens >= compact_threshold:
    → trigger compact (xem buoc 6)
```

### Buoc 6 — Compact (chi khi can)

1. **Mutex per session**: `UPDATE chat_sessions SET compacting=1 WHERE session_id=? AND compacting=0` → chi 1 request duoc compact dong thoi. Tra `0 rows affected` → request khac dang compact → skip, dung raw_turns da truncate (giu 3 gan nhat) lam fallback.
2. **Tach turn**:
   - Giu **3 turn gan nhat** raw (khong dungham).
   - Cac turn cu hon + summary cu (neu co) → input cho compactor.
3. **LLM summarize**: prompt *"Tom tat doan hoi thoai sau thanh 3-5 cau, giu ten dia danh/dia diem/intent chinh, khong them thong tin"*. Model `gpt-4o-mini`, `max_tokens=256`, `temperature=0.2`, retry 3 lan.
4. **Cache**: UPDATE `chat_turns.summary` cua turn dau tien trong range duoc tom tat (hoac INSERT row dac biet `turn_no=0`).
5. **FailGrace**: neu LLM compact fail → fallback dung raw_turns da truncate (giu 3 gan nhat), khong crash chat. Reset `compacting=0`.

### Buoc 7 — Assemble LLM messages

```
[
  {role: system, content: BASE_GUIDE + "\nNGU CANH:\n" + rag_context}
  (neu co summary) {role: user, content: "[TOM TAT LICH SU]: " + summary}
  ...raw_turns (user/assistant theo turn_no)...
  {role: user, content: current_question}
]
```

- `BASE_GUIDE` giu `_SYSTEM_PROMPT` hien co, **bo** cau "Trich dan nguon bang so [1]".
- `max_tokens=800`.

### Buoc 8 — Stream tokens (giu existing SSE)

### Buoc 9 — On done

- UPDATE `chat_turns.answer, intent, tokens_hint` cho turn hien tai.
- UPDATE `chat_sessions.token_total += tokens_hint`.
- Reset `compacting=0` neu da set.

### Vi du thuc te

- Turn 1-4 (chat ngan): ~677 tokens → **chua compact**, gui het 4 turn raw.
- Turn 5-12: dan toi ~2100 → compact, giu 3 turn cuoi, tom tat phan truoc.
- Turn 13+: kiem tra lai, neu 3 turn raw > 2100 → compact tiep (tom tat summary cu + turn giua → summary moi).

---

## 5. Code changes — vi tri cu the

### Backend

| File | Thay doi |
|---|---|
| `src/rag_pipeline/config.py` | Them `MemoryConfig` (enabled, keep_raw_turns=3, compact_threshold=2100, summary_max_tokens=256, summary_temperature=0.2, session_ttl_hours=24). Doi `GenerationConfig.max_tokens=800`. Treo vao `RAGConfig.memory`. **Khong co max_turns.** |
| `src/rag_pipeline/storage/sqlite.py` | `_ensure_schema()` them 2 bang `chat_sessions`, `chat_turns` + `PRAGMA journal_mode=WAL`. |
| `src/rag_pipeline/storage/conversation.py` (MOI) | Class `ConversationStore`: `upsert_session`, `insert_turn`, `update_turn`, `load_recent_turns(session_id)`, `load_summary(session_id)`, `save_summary(session_id, turn_no, summary)`, `acquire_compact_lock(session_id)`, `release_compact_lock(session_id)`, `gc_sessions(ttl_hours)`. |
| `src/rag_pipeline/generation/memory.py` (MOI) | `MemoryTurn` dataclass + `ConversationMemory`: `build_messages(current_question, rag_context, session_id) -> list[dict]`, `est_tokens(text)`, `needs_compact(raw_turns)`, `compute_threshold()` (runtime tu config). |
| `src/rag_pipeline/generation/compactor.py` (MOI) | `MemoryCompactor.compact(session_id, turns_to_summarize, old_summary) -> str`: LLM summarize, retry, cache, mutex acquire/release, fallback. |
| `src/rag_pipeline/generation/answer_generator.py` | Tach method `generate_stream_messages(messages: list[dict])` nhan pre-built list. Giu `generate_stream(query, context)` cu cho demo scripts. Bo dong "Trich dan nguon [1]" khoi `_SYSTEM_PROMPT`. |
| `src/rag_pipeline/generation/rag_pipeline.py` | `answer_stream(query, session_id=None)`: persist user turn → load memory → compact neu can → build messages → stream → persist answer. Backward-compatible: `session_id=None` → flow cu, khong cham DB. |
| `src/rag_pipeline/api/schemas.py` | `ChatRequest`: `question` max_length=500, optional `session_id`. `ChatResponse`/`StreamDone`: them `session_id`, `memory_used`. Bo `sources`/`citation` khoi logic (giu type co the de backward compat). |
| `src/rag_pipeline/api/routes/chat.py` | Inject `ConversationStore` + `MemoryCompactor`; pass `session_id` xuong `pipeline`. |
| `src/rag_pipeline/api/dependencies.py` | Singleton `ConversationStore`, `MemoryCompactor`; inject vao `RAGPipeline`. |

### Frontend

| File | Thay doi |
|---|---|
| `frontend/src/types/index.ts` | `Message` them `turn_no?: number`. `StreamDone`: them `session_id`, `memory_used`. |
| `frontend/src/api/client.ts` | `chatStream(question, sessionId)`: them `session_id` vao body. Helper `getOrCreateSessionId()`: doc `sessionStorage['rag.session_id']`, khong co thi `crypto.randomUUID()`. |
| `frontend/src/hooks/useChat.ts` | Init sessionId tu sessionStorage; moi `sendMessage` truyen kem; nhan `session_id` tu `done` de update ref; `resetSession()` gen id moi + clear messages. |
| `frontend/src/App.tsx` | Nut "Moi" → `resetSession()`. Virtual scroll wrapper cho messages list. |
| `frontend/src/components/MessageBubble.tsx` | Bo phan render `Sources`/`CitationCard`. |
| `frontend/src/components/CitationCard.tsx` | Xoa hoac khong import. |
| `frontend/src/components/ChatInput.tsx` | Them `maxLength={500}` cho textarea + counter ky tu (optional). |

---

## 6. Virtual scroll (Phase 3)

### Research

- **ChatGPT an danh**: mang message trong JS state, render chi ~5-10 bubble tren viewport bang `react-virtuoso`. Moi bubble destroy khi cuon ra khoi viewport → DOM on dinh.
- **Claude (claude.ai)**: dung `TanStack Virtual` cho list; history dai van smooth.

### De xuat cho project

- Thu vien: **@tanstack/react-virtual** (~3KB, khong can stylesheet).
- Wrap `messages.map` o `App.tsx:54` thanh:
  - 1 outer div ref voi height dong.
  - Render chi slice `[start, end]` tu virtualizer.
- Bubble cao dong → dung `measureElement` de tu do cao sau khi render.
- **Fallback khong thu vien**: windowing tay voi `IntersectionObserver` + slice.

---

## 7. Pitfalls / Known issues

1. **Thread-safety SQLite**: `check_same_thread=False` da co; them WAL mode cho concurrent read. Compactor ngam can queue/lock.
2. **Race condition 2 request same session**: FE dung sessionStorage (per-tab) → hiem khi trung. Van de phong: UNIQUE `(session_id, turn_no)` → request thu 2 loi → return 409 hoac auto-retry voi turn_no moi.
3. **`crypto.randomUUID` can HTTPS hoac localhost** → production da co nginx SSL.
4. **Token estimate tieng Viet**: `//3` overestimate ~10-15% → an toan phia tran. Upgrade sang `tiktoken` sau neu can chinh xac.
5. **`max_tokens=800`**: ~400-550 tu tieng Viet — du cho FAQ du lich.
6. **Session GC vs dang stream**: GC check `compacting=0` AND `last_active_at < now - 24h` AND khong co pending turn (`answer IS NULL`). Skip session neu co pending.
7. **Backward compat**: `demo_rag.py`, `benchmark_latency.py` dung `pipeline.answer(query)` khong co session_id → van chay, khong cham DB memory.
8. **Citations [1] [2] nghia doi khi memory nhieu luot**: da bo citation UI → khong con van de. System prompt cung bo rule citation.

---

## 8. Test cases

| File | Scope |
|---|---|
| `tests/conversation/test_store.py` | insert/load/upsert session, conversation_messages, summaries; UNIQUE constraint; lock acquire/release. |
| `tests/conversation/test_memory_assemble.py` | `build_messages` dung thu tu; bo summary khi khong can; truncate theo threshold; est_tokens dung. |
| `tests/conversation/test_compactor.py` | Mocked LLM → summary cached lan 2 khong goi LLM; retry; fallback khi fail; mutex. |
| `tests/api/test_chat_with_memory.py` | POST `/api/chat/stream` voi `session_id` → message persisted + `memory_used=true`. Khong co `session_id` → khong cham DB. |
| `tests/api/test_session_gc.py` | Session cu qua TTL → GC xoa; session dang compact/pending → skip. |
| `tests/api/test_input_output_limits.py` | Input > 500 chars → 422; output khong vuot 800 tokens. |

---

## 9. Thu tu trien khai

### Phase 1 — Backend memory core (chua compact)

- DB schema + WAL mode.
- `ConversationStore` (upsert_session, insert_turn, update_turn, load_recent_turns).
- `ConversationMemory.build_messages` (load tat ca raw, chua compact).
- Validate input 500 chars, output 800 tokens.
- `ChatRequest` them `session_id`, response them `session_id` + `memory_used`.
- Tich hop vao `RAGPipeline.answer_stream` — backward compatible.

### Phase 2 — Compact

- Token estimator `est_tokens`.
- `compute_threshold()` runtime tu config.
- `MemoryCompactor` (LLM summarize, cache, mutex, fallback).
- `needs_compact(raw_turns)` check.
- Tach `keep_raw_turns=3` khi compact.
- Test overflow + cache + fallback.

### Phase 3 — Frontend

- `sessionStorage` session_id, gui kem moi request.
- `resetSession()` cho nut "Moi".
- Bo citation UI (`MessageBubble`, `CitationCard`).
- `ChatInput` maxLength=500 + counter.
- **Virtual scroll** voi @tanstack/react-virtual.

### Phase 4 — GC + polish

- Lazy GC session > 24h.
- 409 race handling (UNIQUE constraint).
- FE retry neu session_id BE tra khac FE gui.
- Badge "Memory on" (optional).
