# RAG Pipeline — Vietnam Tourism

Hệ thống Retrieval-Augmented Generation cho lĩnh vực du lịch Việt Nam. Dự án xây dựng pipeline từ dữ liệu thô đến trả lời tự động, sử dụng dense vector và sparse BM25 vector trong cùng một Qdrant collection.

---

## Dataset

Dữ liệu sử dụng trong dự án là [Vietnam Tourism v2](https://www.kaggle.com/datasets/vuonglsts/vietnam-tourism-v2/data) từ Kaggle. Đây là bộ dữ liệu Hỏi và Đáp tiếng Việt bao gồm nhiều chủ đề về văn hóa, lịch sử, địa điểm, ẩm thực và mẹo du lịch.

Cách dự án sử dụng dataset:

- **Corpus:** Lấy `title` và `context` từ mỗi chủ đề, bỏ qua các cặp câu hỏi-câu trả lời trong `qas`.
- **Evaluation:** Trích xuất `qas` thành tập câu hỏi đánh giá, mỗi chủ đề một câu hỏi đại diện.

Chi tiết xem tại `docs/dataset.md`.

---

## Kiến trúc

Hệ thống được xây dựng theo các tầng rõ ràng:

```text
Source → Document → Chunk → Index
```

### Các thành phần chính

- **Storage Layer:** Quản lý `Source`, `Document`, `Chunk`, `IndexEntry` với UUID.
- **Chunking Pipeline:** Xử lý document qua các giai đoạn Normalize → Clean → Enrich → Section Detect → Chunk → Validate.
- **Embedding:** Tạo dense vector qua OpenRouter và sparse BM25 vector bằng classic BM25 offline.
- **Vector Store:** Lưu trữ và tìm kiếm vector trong Qdrant với cả dense và sparse vectors.
- **Retrieval:** Kết hợp dense search, sparse search, RRF fusion và query preprocessing bằng LLM để trả về kết quả cuối cùng.
- **Generation:** Tạo câu trả lời tiếng Việt có trích dẫn từ các đoạn văn bản đã truy xuất, hỗ trợ streaming.

---

## Tài liệu

- `docs/dataset.md` — Mô tả dataset Vietnam Tourism v2.
- `docs/storage.md` — Kiến trúc tầng lưu trữ.
- `docs/chunking.md` — Chi tiết chunking pipeline.
- `docs/retrieval.md` — Kiến trúc retrieval với hybrid search và query preprocessing.
- `docs/generation.md` — Tạo câu trả lời có trích dẫn và streaming.
- `docs/latency.md` — Kết quả benchmark latency từng đoạn pipeline.

---

## Scripts

- `scripts/ingest_and_index.py` — Ingest dữ liệu và index vào Qdrant.
- `scripts/demo_rag.py` — Chạy demo RAG tương tác với streaming.
- `scripts/benchmark_latency.py` — Đo latency từng đoạn của pipeline và xuất báo cáo JSON/CSV.

---

## Chạy API và Frontend

### Backend API

```bash
$env:PYTHONIOENCODING="utf-8"
python -m rag_pipeline.api.app
```

API chạy tại `http://localhost:8000`, hỗ trợ:
- `GET /api/health` — kiểm tra trạng thái.
- `POST /api/chat` — trả lời không streaming.
- `POST /api/chat/stream` — streaming câu trả lời qua SSE.
- Swagger UI tại `/docs`.

### Frontend Development

```bash
cd frontend
npm install
npm run dev    # → http://localhost:5173
```

Frontend tự động proxy `/api` về backend `http://localhost:8000`.

### Docker Production

```bash
docker compose up -d
```

Image `api` build frontend và serve qua FastAPI trên port `8000`.

---

## Mục tiêu

Xây dựng một RAG pipeline sạch, modular, dễ thay thế từng thành phần, và có khả năng đánh giá chất lượng retrieval trên dataset tiếng Việt.
