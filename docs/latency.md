# Benchmark Latency

Tài liệu này trình bày kết quả đo latency từng đoạn của RAG pipeline du lịch
Việt Nam. Mục tiêu là xác định thời gian được tiêu tốn ở đâu từ lúc nhận
câu hỏi của người dùng đến lúc trả về câu trả lời dạng streaming.

## Phương pháp

Benchmark được chạy bằng `scripts/benchmark_latency.py`, đo độc lập từng giai
đoạn cho mỗi câu hỏi:

- **Tiền xử lý câu hỏi (rewrite):** Viết lại câu hỏi + phân loại intent bằng
  LLM `deepseek/deepseek-v4-flash` qua OpenRouter.
- **Dense embedding:** Tạo vector dày đặc bằng
  `nvidia/llama-nemotron-embed-vl-1b-v2:free` qua OpenRouter.
- **Dense search:** Tìm kiếm vector dày đặc trên Qdrant.
- **Sparse embedding:** Mã hóa BM25 cổ điển chạy local.
- **Sparse search:** Tìm kiếm vector thưa trên Qdrant.
- **Hybrid retrieval đầy đủ:** Dense + sparse + RRF fusion + tra cứu metadata.
- **Xây dựng context:** Định dạng các chunk đã retrieve thành khối context có
  trích dẫn.
- **Generation TTFT:** Thời gian đến token đầu tiên từ `openai/gpt-4o-mini`
  qua OpenRouter.
- **Generation total:** Tổng thời gian sinh câu trả lời.
- **End-to-end:** Tổng thời gian qua toàn bộ `RAGPipeline`.

### Dữ liệu

Ba câu hỏi được lấy từ `data/eval/queries.jsonl`:

1. Người dân Việt Nam nổi tiếng với đặc điểm gì?
2. Nền văn minh sông Hồng có nền tảng từ thời kỳ nào?
3. Bãi biển Nha Trang có đặc điểm gì nổi bật về thời tiết?

### Môi trường

- Qdrant chạy local qua Docker Compose.
- Các API call đến OpenRouter cho dense embedding, query rewriting và answer
  generation.
- SQLite query cache được bật nhưng lạnh cho lần chạy đầu tiên của mỗi câu
  hỏi duy nhất.

## Kết quả

Tất cả thời gian tính bằng millisecond.

| Metric                  |   Mean |    P50 |    P95 |    P99 |
| ----------------------- | -----: | -----: | -----: | -----: |
| rewrite_ms              | 3299.7 | 3229.9 | 5048.7 | 5048.7 |
| dense_embed_ms          |  726.1 |  659.1 |  896.9 |  896.9 |
| dense_search_ms         |   24.1 |   26.6 |   32.0 |   32.0 |
| sparse_embed_ms         |    0.1 |    0.1 |    0.1 |    0.1 |
| sparse_search_ms        |   12.2 |   11.7 |   13.6 |   13.6 |
| retrieve_total_ms       |  737.9 |  677.7 |  891.4 |  891.4 |
| context_build_ms        |    0.0 |    0.0 |    0.1 |    0.1 |
| generation_ttft_ms      | 1522.1 | 1587.9 | 2255.9 | 2255.9 |
| generation_total_ms     | 1990.2 | 1822.2 | 2489.2 | 2489.2 |
| e2e_ms                  | 3597.6 | 2494.0 | 6569.0 | 6569.0 |

## Phân tích

### Những giai đoạn tốn thời gian nhất

Có hai giai đoạn chiếm phần lớn latency:

1. **Tiền xử lý câu hỏi (rewrite): ~3.3s trung bình.** Mô hình DeepSeek V4
   Flash trả về JSON chứa câu hỏi viết lại, intent và độ tin cậy. Đây là API
   call đắt nhất vì mô hình lớn và prompt bao gồm hướng dẫn cùng ví dụ.

2. **Sinh câu trả lời: ~2.0s tổng, ~1.5s TTFT.** GPT-4o-mini có TTFT đáng
   kể, sau đó các token được stream nhanh.

### Phân rã Retrieval

- **Dense retrieval:** ~726ms cho embedding + ~24ms cho search. API embedding
  là phần đắt; tìm kiếm Qdrant rất nhanh.
- **Sparse retrieval:** ~0.08ms mã hóa BM25 local + ~12ms search. So với
  dense retrieval, sparse retrieval gần như miễn phí.
- **RRF fusion và tra cứu metadata:** Thời gian retrieve đầy đủ (~738ms)
   xấp xỉ dense embedding + search, nghĩa là bước fusion thêm overhead không
   đáng kể.

### Các giai đoạn không đáng kể

- **Xây dựng context:** ~0.03ms. Ghép chunk với trích dẫn là thao tác chuỗi
  local đơn giản.
- **Sparse embedding:** ~0.08ms. Tokenization local và tra cứu trọng số BM25.

## Các điểm nghẽn

1. **Tiền xử lý câu hỏi bằng LLM.** Mỗi câu hỏi mất ~3.3s trước khi retrieval
   bắt đầu.
2. **API dense embedding.** Thêm ~726ms và chỉ chạy sau khi câu hỏi đã được
   viết lại.
3. **Generation TTFT.** Người dùng phải chờ ~1.5s trước khi thấy token đầu
   tiên.

## Đề xuất tối ưu

### Ngắn hạn

- **Làm nóng query cache.** Benchmark dùng query lạnh. Trong production, các
  câu hỏi phổ biến sẽ trúng `query_cache` và bỏ qua LLM rewrite hoàn toàn.
- **Cache dense vector cho query thường gặp.** Nếu câu hỏi viết lại lặp lại,
  lưu sẵn dense vector để tránh gọi API embedding.
- **Stream bước rewrite.** Nếu không thể bỏ rewrite, ít nhất hiển thị progress
  event để người dùng cảm thấy chờ đợi ngắn hơn.

### Trung hạn

- **Dùng mô hình rewrite nhỏ hơn / nhanh hơn.** Một mô hình nhẹ hơn như
  `mistralai/mistral-7b-instruct:free` có thể đánh đổi một chút chất lượng
  lấy latency thấp hơn nhiều.
- **Song song hóa dense và sparse retrieval.** Hiện tại pipeline chạy tuần tự;
  dense embedding và sparse encoding có thể chạy song song sau khi có câu
  hỏi viết lại.
- **Giảm generation TTFT.** Dùng mô hình nhanh hơn hoặc rút gọn độ dài context
  gửi đến generator.

### Dài hạn

- **Tự host embedding model.** Chạy dense embedder local loại bỏ latency mạng
  và giới hạn rate limit của API.
- **Sparse + dense index được tính trước.** Đã có sẵn; đảm bảo incremental
  indexing giữ index luôn warm.

## Latency thực tế từ eval

Ngoài benchmark từng giai đoạn, `scripts/eval_rag.py` đo latency end-to-end
thông qua 25 câu hỏi đa dạng (easy, hard, out-of-scope, missing, ambiguous).

Kết quả trên corpus **2.801 chunks / 1.386 documents**:

```text
Avg latency: 10,256ms (~10 giây/câu)
Min: ~5,391ms
Max: ~19,026ms
```

Latency dao động lớn chủ yếu do:
- Rate limit và queue của OpenRouter free tier
- Một số câu generation dài hơn
- Cache lạnh/lạnh không đều

### So sánh với Wiki RAG 2M chunks

Hệ thống trước đây với **~2 triệu chunks** từ Wikipedia tiếng Việt bị chậm
kinh khủng vì:

| Yếu tố | Wiki RAG 2M chunks | Tourism RAG 2.8k chunks |
|--------|-------------------|------------------------|
| Qdrant search | Chậm (index lớn) | Nhanh (~24ms) |
| BM25 index | Có thể rất lớn | Nhỏ, tra cứu local |
| Memory usage | Cao | Thấp |
| Embedding build | Cực kỳ lâu | ~5 phút |

Với tourism RAG, **bottleneck không còn là scale mà là API latency** từ
OpenRouter.

---

## Chiến lược tối ưu chi tiết

### Mục tiêu latency

| Mục tiêu | Thờ i gian | Cách làm |
|----------|-----------|----------|
| Rất tốt | < 3 giây | Self-host embedding + LLM nhỏ |
| Tốt | 3–5 giây | Cache + song song + model rewrite nhỏ hơn |
| Khả thi ngay | 5–7 giây | Cache + song song dense/sparse |

### 1. Cache đa tầng (high impact, dễ làm)

#### Query rewrite cache
`QueryCache` đã tồn tại. Cần đảm bảo cache hit bỏ qua hoàn toàn LLM rewrite.

#### Dense vector cache
Thêm cache trong `DenseEmbedder`:
- Key: normalized query text
- Value: embedding vector
- TTL: vĩnh viễn hoặc 30 ngày

#### Answer cache (tùy chọn)
Với câu hỏi phổ biến, cache luôn câu trả lờ i hoàn chỉnh.

**Hiệu quả dự kiến:** Query trúng cache giảm từ ~10s xuống ~2-3s.

### 2. Song song hóa retrieval (medium impact, dễ làm)

Trong `HybridRetriever.retrieve()`:

```python
# Hiện tại tuần tự
dense_vector = self.dense_embedder.embed([query])[0]
sparse_vector = self.sparse_embedder.embed([query])[0]

# Tối ưu: chạy song song
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    dense_future = executor.submit(self.dense_embedder.embed, [query])
    sparse_future = executor.submit(self.sparse_embedder.embed, [query])
    dense_vector = dense_future.result()[0]
    sparse_vector = sparse_future.result()[0]
```

**Hiệu quả dự kiến:** Giảm ~700ms (bằng thờ i gian sparse embedding gần như 0).

### 3. Tối ưu rewrite (high impact)

#### 3a. Bỏ rewrite cho query đơn giản
Dùng heuristic: nếu query ngắn, không có từ khóa phức tạp, skip LLM rewrite.

```python
if len(query.split()) <= 6 and not needs_expansion(query):
    rewritten = query
```

#### 3b. Dùng model rewrite nhỏ hơn
Thay `deepseek/deepseek-v4-flash` bằng `mistralai/mistral-7b-instruct:free` hoặc
`openai/gpt-4o-mini`. Trade-off: chất lượng rewrite giảm nhẹ.

**Hiệu quả dự kiến:** Giảm 2–3 giây.

### 4. Tối ưu generation

#### 4a. Giảm max_tokens
Hiện tại `max_tokens=1024`. Nếu câu trả lờ i thường ngắn, giảm xuống 512 hoặc
cho phép frontend gửi tham số.

#### 4b. Rút gọn context
Chỉ gửi các chunks có score cao nhất, không gửi toàn bộ top 5.

#### 4c. Dùng model generation nhanh hơn
`gpt-4o-mini` đã khá nhanh. Có thể thử `google/gemini-flash-1.5` free.

### 5. Self-host embedding model (high impact, khó hơn)

Chạy model embedding local, ví dụ:
- `BAAI/bge-m3`
- `Qwen/Qwen3-Embedding-0.6B`

Yêu cầu:
- VPS có GPU hoặc CPU mạnh
- ~2-4 GB VRAM / RAM

**Hiệu quả dự kiến:** Giảm ~700ms + tránh rate limit.

### 6. Async API calls (medium impact)

Chuyển `httpx.Client` sang `httpx.AsyncClient` để:
- Gọi rewrite và embedding song song
- Xử lý nhiều request cùng lúc

Cần refactor pipeline từ sync sang async.

---

## Roadmap đề xuất

### Tuần 1: Quick wins
1. Thêm dense vector cache
2. Song song hóa dense + sparse embedding
3. Review và tối ưu prompt rewrite (ngắn gọn hơn)

### Tuần 2: Rewrite optimization
1. Thêm heuristic skip rewrite
2. Benchmark model rewrite thay thế
3. Cache answer cho top queries

### Tuần 3: Generation optimization
1. Giảm max_tokens mặc định
2. Rút gọn context gửi đến generator
3. Thử model generation nhanh hơn

### Tuần 4: Infrastructure
1. Async pipeline refactor
2. PoC self-host embedding model

---

## Cách chạy lại

### Benchmark per-stage

```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/benchmark_latency.py
```

### Benchmark end-to-end qua eval

```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/eval_rag.py
```

Kết quả được ghi vào:
- `data/benchmarks/latency_<timestamp>.json/.csv`
- `data/eval/report_<timestamp>.json/.csv`
