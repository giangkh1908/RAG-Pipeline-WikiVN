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

## Cách chạy lại

```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/benchmark_latency.py
```

Kết quả được ghi vào `data/benchmarks/latency_<timestamp>.json` và
`data/benchmarks/latency_<timestamp>.csv`.

Để benchmark nhiều query hơn, thay đổi giá trị `limit` trong hàm `main()` của
`scripts/benchmark_latency.py`.
