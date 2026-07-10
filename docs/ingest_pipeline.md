# Ingest Pipeline (v1)

Pipeline index dữ liệu Wikipedia tiếng Việt (JSONL) vào Qdrant với embedding từ OpenRouter.

> **v1**: Batch ingest 1.1M docs, single embedding model, Qdrant vector store.

---

## Mục lục

1. [Cấu hình máy](#1-cấu-hình-máy)
2. [Cài đặt](#2-cài-đặt)
3. [Chuẩn bị](#3-chuẩn-bị)
4. [Chạy index](#4-chạy-index)
5. [Kiểm tra dữ liệu](#5-kiểm-tra-dữ-liệu)
6. [Kiến trúc tổng quan](#6-kiến-trúc-tổng-quan)
7. [Ingest — Đọc dữ liệu](#7-ingest--đọc-dữ-liệu)
8. [Normalize — Chuẩn hóa document](#8-normalize--chuẩn-hóa-document)
9. [Clean — Xóa markup](#9-clean--xóa-markup)
10. [Chunk — Chia nhỏ văn bản](#10-chunk--chia-nhỏ-văn-bản)
11. [Embed — Vector hóa](#11-embed--vector-hóa)
12. [Index — Lưu vào Qdrant](#12-index--lưu-vào-qdrant)
13. [Hiệu năng](#13-hiệu-năng)
14. [Config](#14-config)
15. [Cấu trúc file](#15-cấu-trúc-file)

---

## 1. Cấu hình máy

### Yêu cầu tối thiểu

| Component | Yêu cầu |
|-----------|---------|
| **OS** | Windows 11 / macOS / Linux |
| **CPU** | 4 cores |
| **RAM** | 8 GB |
| **Disk** | 10 GB trống (venv: 264MB, Docker: ~9GB, data: ~2GB) |
| **Network** | Ổn định (gọi OpenRouter API) |

### Cấu hình benchmark

| Component | Chi tiết |
|-----------|---------|
| **OS** | Windows 11 Home Single Language |
| **CPU** | Intel/AMD x86_64 |
| **RAM** | 16 GB |
| **Disk** | SSD NVMe |
| **Docker** | Docker Desktop 4.x với WSL2 backend |

### Disk Usage

| Component | Size | Ghi chú |
|-----------|------|---------|
| `.venv/` | 264 MB | Đã uninstall torch/ML packages |
| `Docker (qdrant)` | ~9 GB | `docker_data.vhdx` (compact định kỳ) |
| `documents/train.jsonl` | 1.7 GB | 1,118,224 dòng |
| `documents/train.jsonl.idx` | 8.5 MB | Offset index (tự tạo) |
| **Qdrant data** | ~4.2 GB | ~434K points (24% data) |
| **Qdrant full 100%** | ~19.5 GB | ~1.79M points (ước tính) |

### Docker Qdrant

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.13.6
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC
    volumes:
      - qdrant_storage:/qdrant/storage
```

**Lưu ý:** `docker_data.vhdx` chỉ tăng, không tự co. Chạy `diskpart → compact vdisk` định kỳ để giải phóng dung lượng.

---

## 2. Cài đặt

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -e ".[indexing]"
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[indexing]"
```

### Packages chính

| Package | Size | Purpose |
|---------|------|---------|
| `qdrant-client` | ~5 MB | Qdrant API client |
| `httpx` | ~2 MB | HTTP client cho OpenRouter |
| `python-dotenv` | ~0.5 MB | Load .env file |

**Không cần** `torch`, `sentence-transformers`, `numpy` — RecursiveChunker chạy bằng stdlib.

---

## 3. Chuẩn bị

### Windows

```powershell
docker compose up -d
copy .env.example .env
```

### macOS / Linux

```bash
docker compose up -d
cp .env.example .env
```

Sửa file `.env`: điền `OPENROUTER_API_KEY` (lấy tại https://openrouter.ai/keys)

---

## 4. Chạy index

Luôn chạy từ thư mục gốc của project (không `cd src`):

```powershell
# Fresh run — xóa data cũ, nhanh nhất
python -m rag_pipeline.main --clear

# Incremental — giữ data cũ, skip đã index
python -m rag_pipeline.main

# Test nhỏ trước
python -m rag_pipeline.main --clear --sample 0.05
python -m rag_pipeline.main --clear --sample 1
```

### CLI flags

| Flag | Default | Ý nghĩa |
|------|---------|---------|
| `--clear` | `false` | Xóa Qdrant collection trước khi chạy. Bỏ check Qdrant → chỉ dùng `seen` set trong RAM → **nhanh hơn 100x** |
| `--sample` | `100.0` | Phần trăm data cần index (0.05 = 0.05%) |
| `--qdrant` | `true` | Dùng Qdrant (false = InMemory cho test) |

---

## 5. Kiểm tra dữ liệu

```powershell
$env:PYTHONPATH="src"; python -c "
from qdrant_client import QdrantClient
c = QdrantClient('http://localhost:6333')
info = c.get_collection('wikipedia_vi_chunks')
print(f'Total chunks: {info.points_count}')
"
```

### Test retrieval

```powershell
$env:PYTHONPATH="src"; python -c "
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
from qdrant_client import QdrantClient
from rag_pipeline.config import EmbeddingConfig
from rag_pipeline.indexing.embedder import OpenRouterEmbeddingClient

c = QdrantClient('http://localhost:6333')
embedder = OpenRouterEmbeddingClient(EmbeddingConfig())
query = 'Thủ duong cua Viet Nam o dau?'
vector = embedder.embed_texts([query])[0]
results = c.search(collection_name='wikipedia_vi_chunks', query_vector=('dense', vector), limit=3)
for i, hit in enumerate(results):
    print(f'{i+1}. [{hit.score:.4f}] {hit.payload.get(\"title\", \"?\")}')
    print(f'   {hit.payload.get(\"source_url\", \"\")}')
    print(f'   {hit.payload.get(\"text\", \"\")[:200]}...')
    print()
"
```

---

## 6. Kiến trúc tổng quan

Pipeline chạy theo cơ chế **background flush** — main thread đọc docs, background thread embed + upsert song song:

```
Main Thread                          Background Thread
───────────                          ─────────────────
┌──────────┐   ┌─────────┐   ┌──────┐
│  Read    │──▶│Normalize│──▶│Chunk │──┐
│  JSONL   │   │ + Clean │   │      │  │  buffer
└──────────┘   └─────────┘   └──────┘  ├──▶ [500 chunks] ──┐
                                        │                   ▼
┌──────────┐   ┌─────────┐   ┌──────┐  │    ┌─────────────────────┐
│  Read    │──▶│Normalize│──▶│Chunk │──┘    │ Embed (4 workers    │
│  next... │   │ + Clean │   │      │      │ parallel) + Batch   │
└──────────┘   └─────────┘   └──────┘      │ upsert Qdrant       │
                                            └─────────────────────┘
← đọc tiếp, KHÔNG CHỜ embed
```

### Tại sao nhanh?

| Tối ưu | Trước | Sau | Speedup |
|--------|-------|-----|---------|
| **Background flush** | Đọc → chờ embed → đọc tiếp | Đọc + embed song song | ~2x |
| **Batch upsert** | 500 calls Qdrant/batch | 1 call Qdrant/batch | ~34x |
| **Bỏ Qdrant check** (`--clear`) | 1 scroll query/doc | 0 query (chỉ `seen` set) | ~100x |
| **Parallel workers** | 1 API call/lần | 4 API calls song song | ~4x |

---

## 7. Ingest — Đọc dữ liệu

### `LocalJsonlReader` — Offset Index

**Vấn đề:** File `train.jsonl` nặng **1.7GB** (1.1M dòng). Khi sample nhỏ, vẫn đọc toàn bộ file → chậm.

**Giải pháp:** Offset index — lần đầu scan file tạo `.idx`, lần sau seek trực tiếp.

#### Cấu trúc file index

```
train.jsonl.idx (8.5MB cho 1.1M records)

┌─────────────────────────────────────────┐
│ Header (16 bytes)                       │
│  - Magic: "RGOF" (4B)                  │
│  - Version: 1 (4B, uint32)             │
│  - Count: 1118224 (8B, uint64)         │
├─────────────────────────────────────────┤
│ Entry 0: byte_offset (8B, uint64)      │
│ Entry 1: byte_offset (8B, uint64)      │
│ ...                                     │
│ Entry 1118223: byte_offset (8B)        │
└─────────────────────────────────────────┘
```

#### Hiệu năng

| Scenario | Không index | Có index | Speedup |
|----------|-------------|----------|---------|
| 0.01% sample, lần 1 | 7.7s | 3.3s | 2x |
| 0.01% sample, lần 2+ | 7.7s | **0.3s** | **25x** |

### Các Reader khác

| Reader | File | Mục đích |
|--------|------|----------|
| `HuggingFaceDatasetReader` | — | Load từ HuggingFace Hub |
| `LocalCorpusCsvReader` | `corpus.csv` | Đọc corpus CSV (cột `cid`, `text`) |
| `LocalQueryCsvReader` | `train.csv`, `val.csv` | Đọc query CSV cho evaluation |

---

## 8. Normalize — Chuẩn hóa document

`UVWWikipediaDocumentNormalizer` chuyển `SourceRecord.payload` → `CanonicalDocument`:

```python
# Input
{"id": "Viet_Nam", "title": "Việt Nam", "content": "...", "quality": 9}

# Output
CanonicalDocument(
    doc_id="a1b2c3d4...",       # SHA256 hash (24 chars)
    source_id="Viet_Nam",
    title="Việt Nam",
    checksum="a1b2c3d4...",     # Full SHA256 — dùng cho dedup
    ...
)
```

**Dedup:** Cùng content → cùng checksum → `seen` set bắt trùng trong cùng 1 run.

---

## 9. Clean — Xóa markup

### `WikipediaArticleCleaner` v2

`WikipediaArticleCleaner` xử lý text qua 6 bước:

```
Raw Wikipedia text
    │
    ▼
① _remove_markup_residue()   ← {{template}}, <ref>, <html>, __NOTOC__
    │
    ▼
② _unwrap_wikilinks()        ← [[target|display]] → display
    │
    ▼
③ _strip_formatting()        ← '''bold''' → bold, ''italic'' → italic
    │
    ▼
④ _remove_template_artifacts() ← | key = value, }}
    │
    ▼
⑤ _repair_broken_lines()     ← nối câu bị ngắt dòng
    │
    ▼
⑥ Normalize whitespace       ← nhiều space → 1 space, nhiều \n → \n\n
    │
    ▼
Clean text
```

### Cải tiến v2

| Vấn đề v1 | Fix v2 |
|---|---|
| Template lồng nhau `{{A|{{B|C}}}}` bị cắt sai | Lặp bóc template trong cùng nhất đến khi hết |
| Infobox value tiếp tục sang dòng mới (`\| key = value\n| other = ...`) bị sót | Regex multi-line remove infobox continuations |
| Các dòng remnant như `(Hà Nội) (Huế) ~ (TP. HCM)` còn lại | Giảm ~25% noise trên sample 30 bài |

Chi tiết kiến trúc clean: xem `src/rag_pipeline/transform/cleaner.py` và `tests/test_cleaner.py`.

---

## 10. Chunk — Chia nhỏ văn bản

### `StructuredChunker` — Chiến lược chia v2

> **v2 thay thế `RecursiveChunker` bằng `StructuredChunker`.**
> Chi tiết: [docs/chunking.md](chunking.md)

Pipeline mới parse cấu trúc bài viết trước khi chia:

```
Clean text
    │
    ▼
Parse blocks (heading / paragraph / list)
    │
    ▼
Heading = hard boundary
    │
    ▼
Gom blocks cùng section thành chunk
    │
    ▼
Thêm Anthropic-style context prefix
    │
    ▼
DocumentChunk.text = "{context}\n\n{raw_text}"
```

### Tại sao đổi?

| | RecursiveChunker (v1) | StructuredChunker (v2) |
|---|---|---|
| Boundary | Paragraph / sentence | Heading (hard boundary) |
| Section awareness | Không | Có (`section_path`) |
| Context prefix | Không | Natural language |
| List handling | Có thể cắt ngang | Giữ nguyên list |
| Reference sections | Không phân biệt | `is_reference_section` flag |
| Dependencies | stdlib | stdlib |

### Tại sao không dùng SemanticChunking?

| | SemanticChunker | StructuredChunker |
|---|---|---|
| Dependencies | `sentence-transformers`, `torch`, `numpy` | Không có |
| 1.1M docs | ~30 giờ | ~vài phút |
| Chất lượng | Semantic coherence cao | Giữ section boundary + context |
| Disk usage | ~5 GB (torch) | 0 MB |

---

## 11. Embed — Vector hóa

### `OpenRouterEmbeddingClient` — Parallel Sub-batch + Retry

**Model:** `nvidia/llama-nemotron-embed-vl-1b-v2:free`
- Context window: **131,072 tokens**
- Output: **2048-dim vectors**
- Pricing: **$0** (free tier)
- https://openrouter.ai/nvidia/llama-nemotron-embed-vl-1b-v2:free
### Parallel Workers

```
Workers  1:  61 texts/s   (sequential)
Workers  2:  97 texts/s   (1.6x)
Workers  4: 249 texts/s   (4x) ✅ tối ưu
Workers  6: 242 texts/s   (không tăng — bottleneck là API rate limit)
```

### Retry on 429

```python
for attempt in range(max_retries + 1):
    response = client.post("/embeddings", json=payload)

    if response.status_code == 429:           # Rate limited
        delay = 2.0 * (2 ** attempt)          # 2s → 4s → 8s
        time.sleep(delay)
        continue

    response.raise_for_status()
    return response.json()["data"]
```

---

## 12. Index — Lưu vào Qdrant

### Batch Upsert

**Trước:** Mỗi document = 1 lần upsert → 500 calls/batch
**Sau:** Gom tất cả points trong batch → 1 lần upsert

```python
# Gom tất cả points từ 500 chunks
all_points = [PointStruct(...), PointStruct(...), ...]  # 500 points

# 1 call duy nhất
client.upsert(collection_name="wikipedia_vi_chunks", points=all_points)
```

### Vector Structure

```python
PointStruct(
    id="a1b2c3d4-e5f6-...",   # UUID từ chunk_id
    vector={"dense": [0.12, -0.34, ...]},  # 2048-dim
    payload={
        "doc_id": "doc-123",
        "text": "Việt Nam là quốc gia...",
        "title": "Việt Nam",
        "source_url": "https://vi.wikipedia.org/wiki/Việt_Nam",
        "chunk_index": 0,
        "prev_chunk_id": null,
        "next_chunk_id": "chunk-2",
        ...
    }
)
```

### Idempotent

- `--clear`: Xóa collection, bỏ check Qdrant → chỉ dùng `seen` set trong RAM → **nhanh nhất**
- Không `--clear`: Check `has_document_version()` cho mỗi doc → chậm hơn khi collection lớn

---

## 13. Hiệu năng

### Benchmark thực tế

**Dataset:** `documents/train.jsonl` — 1.7 GB, 1,118,224 documents, ~1.79M chunks

| Metric | Giá trị |
|--------|---------|
| Tốc độ ingest (main thread) | **1,810 docs/s** |
| Tốc độ embed (4 workers) | **249 texts/s** |
| Full 100% — main thread | ~10 phút |
| Full 100% — background upsert | ~5-6 giờ (tùy API speed) |
| 1 batch (500 chunks) | ~8s |
| Offset index speedup | 25x on repeat reads |
| Peak memory | ~550 MB |

### So sánh qua các lần tối ưu

| Version | Tốc độ | Full 100% | Cải thiện |
|---------|--------|-----------|-----------|
| SemanticChunker + sequential | 5 docs/s | ~62 giờ | 1x |
| RecursiveChunker + batch 500 | 34 docs/s | ~9 giờ | 7x |
| + Background flush | 45 docs/s | ~7 giờ | 9x |
| + Batch upsert Qdrant | 45 docs/s | ~7 giờ | 9x |
| + Bỏ Qdrant check (`--clear`) | **1,810 docs/s** | **~10 phút** | **~360x** |

### Test runs đã thực hiện

| Run | Sample | Docs | Points | Thời gian | Tốc độ |
|-----|--------|------|--------|-----------|--------|
| Test 0.01% | 0.01% | 112 | — | 0.08s | 1,400 docs/s |
| Test 0.05% | 0.05% | 560 | 832 | 34s | 16 docs/s |
| Test 1% | 1% | 11,181 | — | 6s | 1,864 docs/s |
| Full (background) | 100% | 1,118,000 | 434,500 | 618s + ~5 giờ | 1,810 docs/s |

### Progress Log

```
📊 1,118,000 docs | 1,117,796 indexed |    204 skip | 1,790,235 chunks | 3580 batches |    618s |    1810 docs/s
```

| Field | Ý nghĩa |
|-------|---------|
| `docs` | Tổng docs đã xử lý |
| `indexed` | Docs đã embed + upsert thành công |
| `skip` | Docs bị bỏ qua (trùng checksum trong `seen` set) |
| `chunks` | Tổng chunks đã tạo |
| `batches` | Số lần gọi embed API |
| `docs/s` | Tốc độ xử lý (main thread) |

---

## 14. Config

| Config | Default | Ý nghĩa |
|--------|---------|---------|
| **Chunking** | | |
| `chunking.max_tokens_per_chunk` | 300 | Số token tối đa cho **toàn bộ** input embedding (context + body) |
| `chunking.chunk_overlap_tokens` | 40 | Số token overlap giữa chunk liền kề |
| `chunking.min_chunk_tokens` | 40 | Ngưỡng merge chunk đuôi nhỏ trong cùng section |
| `chunking.estimated_context_tokens` | 40 | Margin an toàn cho tokenizer thực tế |
| **Embedding** | | |
| `embedding.model_name` | `nvidia/llama-nemotron-embed-vl-1b-v2:free` | Model trên OpenRouter |
| `embedding.api_base` | `https://openrouter.ai/api/v1` | API endpoint |
| `embedding.sub_batch_size` | 500 | Số texts mỗi API call |
| `embedding.max_retries` | 3 | Số lần retry khi 429 |
| `embedding.parallel_workers` | 4 | Số API calls song song |
| `embedding.timeout_seconds` | 30 | Timeout mỗi request |
| **Qdrant** | | |
| `qdrant.url` | `http://localhost:6333` | Qdrant server |
| `qdrant.collection_name` | `wikipedia_vi_chunks` | Collection name |
| `qdrant.dense_vector_name` | `"dense"` | Tên dense vector field |
| **Pipeline** | | |
| `embed_batch_size` | 500 | Số chunks tích lũy trước khi flush |
| `flush_workers` | 2 | Số background thread flush |

---

## 15. Cấu trúc file

| File | Chức năng |
|------|-----------|
| `main.py` | Entry point — `python -m rag_pipeline.main` |
| `config.py` | Config dataclasses |
| `models.py` | Data models: `SourceRecord`, `CanonicalDocument`, `DocumentChunk`, `IndexedChunk` |
| `ingest/dataset.py` | `LocalJsonlReader` (offset index), `LocalCorpusCsvReader`, `LocalQueryCsvReader`, `HuggingFaceDatasetReader` |
| `ingest/normalize.py` | `UVWWikipediaDocumentNormalizer` — payload → `CanonicalDocument` |
| `transform/cleaner.py` | `WikipediaArticleCleaner` — 6-step wiki markup removal |
| `transform/structure_chunker.py` | `StructuredChunker` — heading-aware chunks + Anthropic-style context |
| `pipelines/ingest_pipeline.py` | `IngestPipeline.run()` — background flush, parallel embed, batch upsert |
| `indexing/embedder.py` | `OpenRouterEmbeddingClient` (parallel sub-batch + retry), `DeterministicTestEmbedder` |
| `indexing/vector_store.py` | `QdrantVectorStore` (batch upsert), `InMemoryVectorStore` |
| `utils/hashing.py` | `stable_hash()` — SHA256 deterministic hash |
