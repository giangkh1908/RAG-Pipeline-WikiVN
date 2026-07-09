# Chunking Architecture (v2)

Kiến trúc chunking mới cho RAG Pipeline WikiVN — thay thế `RecursiveChunker` bằng `StructuredChunker` để cải thiện chất lượng retrieval.

> **Mục tiêu v2:** chunk có ý nghĩa ngữ nghĩa, giữ boundary đúng cấu trúc bài viết, và được đóng gói context để embedding model hiểu "đây là đoạn nào của bài nào".

---

## 1. Tổng quan

Pipeline cũ (v1) dùng `RecursiveChunker`: chia text theo paragraph → câu → word window. Nhanh, không cần dependency, nhưng:
- Không biết đâu là heading, đâu là nội dung
- Cắt ngang section, gộp lộn section khác nhau
- Không có context → embedding khó biết chunk thuộc chủ đề gì

Pipeline mới (v2) dùng `StructuredChunker`:
- Phân tích cấu trúc bài viết (heading / paragraph / list)
- Heading = hard boundary
- Giữ list items trong cùng một chunk
- Gán section path (`Title > Section > Subsection`)
- Sinh **context prefix** theo phong cách Anthropic cho mỗi chunk
- Đánh dấu reference section (`Tham khảo`, `Liên kết ngoài`, ...)
- Giữ liên kết prev/next chunk cho retrieval expansion

---

## 2. Data flow

```
CanonicalDocument (title + cleaned content)
        │
        ▼
┌─────────────────────┐
│  _parse_blocks()    │  ← chia thành Block{kind, text, level, section_path}
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ _extract_doc_summary()│  ← lấy câu đầu tiên có ý nghĩa làm document summary
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ _group_into_chunks() │  ← gom blocks thành chunk, split khi vượt max_tokens
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  _build_context()    │  ← tạo Anthropic-style context
└─────────────────────┘
        │
        ▼
DocumentChunk.text = "{context}\n\n{raw_text}"
```

---

## 3. Block parsing

Cleaned text được tách thành các đoạn (`\n\n+`). Mỗi đoạn được phân loại:

| Kind | Nhận diện | Ví dụ |
|------|-----------|-------|
| `heading` | 1 dòng ngắn, bắt đầu hoa, không kết thúc bằng dấu câu, không phải URL/list | `Lịch sử`, `Địa lý`, `Thân thế` |
| `list` | Tất cả các dòng đều bắt đầu bằng list marker | `- item 1\n- item 2` |
| `paragraph` | Còn lại | Đoạn văn thường |

### Heading detection heuristic

Một dòng được coi là heading nếu:
- Không quá 10 từ / 100 ký tự
- Không bắt đầu bằng `*`, `#`, `-`, số thứ tự
- Không kết thúc bằng `. ! ? ; :`
- Không phải URL
- Bắt đầu bằng chữ hoa hoặc dấu ngoặc mở
- Không giống infobox remnant: `(Hà Nội) (Huế) ~ (TP. HCM)`

### Heading level estimation

```
Level 1: major section → Lịch sử, Địa lý, Thân thế, Kinh tế, Văn hóa, ...
Level 2: sub-section  → Dòng dõi, Thờ i kỳ, Các loài, Năm 1945, ...
```

Section path được cập nhật theo level:
```python
section_path = section_path[:level] + [heading_text]
# Ví dụ: ["Việt Nam", "Lịch sử", "Thờ i kỳ Bắc thuộc"]
```

---

## 4. Chunk grouping

Quy tắc gom block thành chunk:

1. Khi gặp heading → flush chunk hiện tại, bắt đầu chunk mới với section path mới
2. Paragraph/list liên tiếp cùng section được gom chung nếu tổng token ≤ `max_tokens_per_chunk`
3. Nếu block đơn lẻ vượt quá `max_tokens_per_chunk` → tự thành 1 chunk (hiện tại không split thêm)

```python
groups: list[tuple[str, list[str], bool]]
# (raw_text, section_path, is_reference_section)
```

---

## 5. Context prefix (Anthropic-style)

Mỗi chunk được gắn context tự nhiên thay vì bracket markup:

```text
This chunk is from the 'Việt Nam' document, which describes: Việt Nam, tên gọi chính thức là Cộng hòa Xã hội chủ nghĩa Việt Nam, là một quốc gia nằm ở cực Đông của bán đảo Đông Dương thuộc khu vực Đông Nam Á., specifically the 'Lịch sử > Thờ i kỳ Bắc thuộc' section.

Nội dung đoạn văn bản thực tế ở đây...
```

### Tại sao dùng natural language?

- Embedding model (như Qwen3-Embedding) được train trên câu tự nhiên
- `Title > Section > Subsection` dạng bracket ít mang semantic bằng câu đầy đủ
- Context giúp phân biệt chunk cùng chủ đề ở các bài khác nhau

### Build context

```python
def _build_context(title, doc_summary, section_path):
    ctx = f"This chunk is from the '{title}' document"
    if doc_summary:
        ctx += f", which describes: {doc_summary[:120]}"
    if len(section_path) > 1:
        sec = " > ".join(section_path[1:])
        ctx += f", specifically the '{sec}' section"
    return ctx + "."
```

---

## 6. Chunk text format

```python
full_text = f"{context}\n\n{raw_text}"
```

- `context`: natural-language context (dòng 1)
- `raw_text`: nội dung gốc của chunk (các dòng sau)
- Phân cách bằng `\n\n` để dễ tách lại

Helper có sẵn:
```python
context, text = StructuredChunker.split_context_and_text(chunk.text)
```

JSON output cũng lưu `context` và `text` riêng:
```json
{
  "context": "This chunk is from the 'Việt Nam' document...",
  "text": "Nội dung đoạn văn...",
  "section_path": ["Việt Nam", "Lịch sử", "Thờ i kỳ Bắc thuộc"],
  "is_reference_section": false
}
```

---

## 7. Reference section handling

Các heading sau được đánh dấu `is_reference_section = true`:

- `Tham khảo`, `Chú thích`, `Tài liệu tham khảo`
- `Liên kết ngoài`, `Xem thêm`, `Đọc thêm`
- `Ghi chú`, `Chú giải`, `Trích dẫn`
- `References`, `See also`, `External links`, ...

Ý nghĩa: retrieval có thể ưu tiên thấp hơn hoặc filter bỏ các chunk này vì thường chỉ chứa link/footnote, ít giá trị trả lờ i.

---

## 8. Neighbor linking

Mỗi `DocumentChunk` có:
- `prev_chunk_id`: chunk liền trước trong cùng document
- `next_chunk_id`: chunk liền sau trong cùng document

Dùng cho **context expansion** lúc retrieval: khi lấy 1 chunk, có thể lấy thêm prev/next để LLM có context rộng hơn.

---

## 9. So sánh với v1

| | RecursiveChunker (v1) | StructuredChunker (v2) |
|---|---|---|
| Dependencies | stdlib | stdlib |
| Speed | ~22s / 1.1M docs | Tương đương (regex + heuristics) |
| Boundary | Paragraph / sentence | Heading (hard boundary) |
| Section awareness | Không | Có (`section_path`) |
| Context prefix | Không | Anthropic-style natural language |
| List handling | Có thể cắt ngang | Giữ nguyên list |
| Reference sections | Không phân biệt | `is_reference_section` flag |
| Prev/next links | Có | Có |
| Token count | Theo config | Theo config |

---

## 10. Config

```python
class ChunkingConfig:
    max_tokens_per_chunk: int = 300
```

Mỗi chunk cuối cùng sẽ lớn hơn 300 tokens một chút do context prefix được thêm vào. Điều này là chấp nhận được vì embedding model có context window lớn.

---

## 11. Files

- `src/rag_pipeline/transform/structure_chunker.py` — implementation
- `src/rag_pipeline/transform/chunker.py` — RecursiveChunker (legacy, giữ lại cho test/reference)
- `tests/test_structure_chunker.py` — 22 tests
- `tests/test_chunker.py` — 13 tests cho RecursiveChunker (legacy)
- `docs/chunking.md` — file này

---

## 12. Known limitations

- Heading detection là heuristic — văn bản không theo cấu trúc Wikipedia chuẩn có thể bị miss/sai level
- Chưa split paragraph đơn lẻ quá dài (hiếm gặp sau clean)
- Clean text phải đủ tốt để paragraph/heading phân tách rõ ràng
