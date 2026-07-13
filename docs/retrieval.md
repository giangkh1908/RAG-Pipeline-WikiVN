# Retrieval

Tầng retrieval chịu trách nhiệm tìm kiếm các đoạn văn bản (chunks) liên quan đến câu hỏi của người dùng từ Qdrant. Hệ thống sử dụng kết hợp tìm kiếm ngữ nghĩa (dense) và tìm kiếm từ khóa (sparse), sau đó hợp nhất kết quả bằng Reciprocal Rank Fusion.

---

## Kiến trúc tổng quan

```text
Raw query
    ↓
Query Preprocessing
    - Normalize
    - LLM Rewrite
    - Intent Classification
    - Filter Building
    ↓
Hybrid Search
    - Dense search (OpenRouter embedding)
    - Sparse search (Classic BM25)
    ↓
RRF Fusion
    ↓
Ranked RetrievalResult[]
```

---

## Phase 1: Hybrid Search Core

### Dense Search

Sử dụng vector embedding dày đặc từ OpenRouter với model `nvidia/llama-nemotron-embed-vl-1b-v2:free`. Câu hỏi được embed thành vector 2048 chiều, sau đó tìm kiếm cosine similarity trong Qdrant.

### Sparse Search

Sử dụng Classic BM25 offline, không cần tải model từ HuggingFace. Tầng này xây dựng vocabulary và tính toán IDF từ toàn bộ corpus chunks, sau đó mã hóa câu hỏi thành sparse vector. Vocabulary được persist vào `data/bm25_vocab.json` để tái sử dụng khi query.

### RRF Fusion

Reciprocal Rank Fusion kết hợp thứ hạng từ dense và sparse search:

```text
score(chunk) = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
```

Trong đó `k` là hyperparameter, mặc định là 60. Các chunk xuất hiện trong cả hai kết quả sẽ được ưu tiên cao hơn.

### RetrievalResult

Mỗi kết quả trả về bao gồm:
- `chunk_id`: định danh UUID của chunk
- `content`: nội dung văn bản đầy đủ
- `rrf_score`: điểm sau khi hợp nhất
- `rank`: thứ hạng cuối cùng
- `dense_score`: điểm từ dense search (nếu có)
- `sparse_score`: điểm từ sparse search (nếu có)
- `metadata`: payload từ Qdrant bao gồm title, source_id, section_path, v.v.

---

## Phase 2: Query Preprocessing

### Normalize

Chuẩn hóa câu hỏi cơ bản: chuyển thành chữ thường, loại bỏ khoảng trắng thừa.

### LLM Rewrite

Sử dụng DeepSeek V4 Flash qua OpenRouter để viết lại câu hỏi cho phù hợp với retrieval. Quá trình này giúp:
- Mở rộng từ viết tắt và thuật ngữ mơ hồ
- Thêm từ khóa du lịch khi cần
- Giữ nguyên ngôn ngữ tiếng Việt

Prompt được viết bằng tiếng Việt để phù hợp với corpus.

### Intent Classification

Cùng một LLM call với rewrite, hệ thống phân loại intent thành một trong các loại:
- `factual`: hỏi thông tin thực tế
- `recommendation`: đề xuất, gợi ý
- `comparison`: so sánh
- `list`: danh sách
- `procedural`: hướng dẫn
### Filter Builder

Dựa vào intent, hệ thống xây dựng Qdrant payload filters. Mặc định, các intent phổ biến sẽ loại trừ các reference sections để tránh kết quả không liên quan.

### Query Cache

Kết quả rewrite và intent được cache trong SQLite table `query_cache` để tránh gọi LLM lặp lại cho cùng một query. Cache key phụ thuộc vào model name, prompt version, và raw query text, giúp tự động invalidate khi model hoặc prompt thay đổi.

Nếu LLM call thất bại, hệ thống fallback về normalized query với intent mặc định là `factual`.

---

## RetrievalPipeline

`RetrievalPipeline` là wrapper cấp cao điều phối toàn bộ quy trình:

```text
query → QueryPreprocessor → FilterBuilder → HybridRetriever → RetrievalResult[]
```

Pipeline giúp tách biệt rõ ràng giữa preprocessing và search, cho phép dễ dàng thay thế hoặc mở rộng từng thành phần trong tương lai.

---

## Cấu hình

Các tham số chính trong `RetrievalConfig`:

- `qdrant.dense_top_k`: số kết quả dense search
- `qdrant.sparse_top_k`: số kết quả sparse search
- `rrf_k`: hyperparameter của RRF
- `rrf_top_k`: số kết quả cuối cùng
- `llm_query.model_name`: model LLM cho rewrite/intent
- `llm_query.prompt_version`: version prompt để cache invalidation
- `llm_query.cache_ttl_days`: thời hạn cache

---

## Payload trong Qdrant

Mỗi point trong Qdrant lưu payload giàu metadata:

- `document_id`: UUID của document
- `chunk_order`: thứ tự chunk trong document
- `title`: tiêu đề topic/document
- `source_id`: UUID của source
- `section_path`: đường dẫn section
- `is_reference_section`: cờ đánh dấu section tài liệu tham khảo

Payload này hỗ trợ filtering và hiển thị nguồn cho câu trả lời.

---

## Hạn chế hiện tại

- Tokenizer của Classic BM25 đơn giản, có thể gây false positive khi từ trong query là substring của từ khác (ví dụ: "sánh" trong "so sánh" match với "sánh bước").
- Filter Builder còn đơn giản, chủ yếu loại trừ reference sections.
- Intent classification phụ thuộc vào chất lượng output JSON của LLM.

---

## Hướng phát triển

Các cải tiến có thể bổ sung trong tương lai:
- Thêm cross-encoder reranker để cải thiện thứ hạng
- Áp dụng MMR để tăng tính đa dạng kết quả
- Deduplicate chunks từ cùng document
- Cải thiện tokenizer BM25 cho tiếng Việt
- Filter Builder nâng cao theo title và section
